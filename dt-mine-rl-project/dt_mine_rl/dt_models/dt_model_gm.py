import torch
import torch.nn as nn
import numpy as np

import os
import safetensors.torch
from dt_models.dt_model_common import AgentDT

from dt_models.dt_model_common import ActionPolicy, ConfigActionPolicy, AgentDT, GlobalPositionEncoding, MultiCategorical
from lib.common import AGENT_DT_ACTION_DIM, AGENT_DT_NUM_CAMERA_ACTIONS, AGENT_DT_NUN_ESC_BUTTON

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

SAFE_WEIGHTS_NAME = "model.safetensors"
class MaskedSelfAttention(nn.Module):

    def __init__(self, h_dim, num_heads, seq_len, dropout):
        super().__init__()

        self.h_dim = h_dim # embeding dimensionality, includes all heads
        self.seq_len = seq_len
        self.num_heads = num_heads #  num heads
        assert self.h_dim % self.num_heads == 0 , \
            "Embedding dimension must be multiple of the number of heads."

        # key, query, value projections
        self.proj_q = nn.Linear(h_dim, h_dim)
        self.proj_k = nn.Linear(h_dim, h_dim)
        self.proj_v = nn.Linear(h_dim, h_dim)

        # output projection
        self.proj_out = nn.Linear(self.h_dim, self.h_dim)
        # regularization
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x):

        B, T, C = x.shape # batch size, sequence length, (h_dim * n_heads)
        N, D = self.num_heads, C // self.num_heads # num heads, attenion dim

        # calculate query, key, values
        q = self.proj_q(x).view(B, T, N, D).transpose(1, 2) # (B, seqLen, numHeads, headSize) -> (B, numHeads, seqLen, headSize)
        k = self.proj_k(x).view(B, T, N, D).transpose(1, 2) # (B, seqLen, numHeads, headSize) -> (B, numHeads, seqLen, headSize)
        v = self.proj_v(x).view(B, T, N, D).transpose(1, 2) # (B, seqLen, numHeads, headSize) -> (B, numHeads, seqLen, headSize)

        # causal self-attention; Self-attend: (B, numHeads, seqLen, headSize) x (B, numHeads, headSize, seqLen) -> (B, numHeads, seqLen, seqLen)
        # scaled_dot_product
        attn_logits = (q @ k.transpose(-2, -1))
        attn_logits = attn_logits / torch.sqrt(torch.tensor(k.size(-1)))
        # (B, N, T, T)
        # apply mask
        #mask = torch.zeros(T, x.shape[0]).bool()
        subsequent_mask = torch.triu(input=torch.ones(T, T), diagonal=1).bool() 
        selfattn_mask = subsequent_mask.to(device) # + padding mask
        attn_logits = attn_logits.masked_fill(selfattn_mask, float('-inf'))

        softmax = nn.Softmax(dim=-1)
        attention = softmax(attn_logits)

        attention = attention @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        out = self.attn_dropout(attention)

        out = out.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.proj_out(out))
        return y

class MLP(nn.Module):

    def __init__(self, h_dim, mlp_ratio, dropout):
        super().__init__()
        self.fc1 = nn.Linear(h_dim, mlp_ratio*h_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(mlp_ratio*h_dim, h_dim)
        self.drop= nn.Dropout(dropout)

    def forward(self, x):

        x = self.act(self.fc1(x))
        x = self.drop(self.fc2(x))
        return x

class DecoderBlock(nn.Module):

    def __init__(self, h_dim, num_heads, seq_len, mlp_ratio, dropout):
        super().__init__()

        self.attn = MaskedSelfAttention(h_dim, num_heads, seq_len, dropout)
        self.ln1 = nn.LayerNorm(h_dim)
        self.mlp = MLP(h_dim, mlp_ratio, dropout)
        self.ln2 = nn.LayerNorm(h_dim)

    def forward(self, x):
        """ x = self.ln2(x) # add residual
        x = self.attn(x) + x # normalize
        x = self.ln2(x)
        x = self.mlp(x) + x """

        attn_out= self.attn(x)# add residual
        x = self.ln1(x + attn_out)
        mlp_out = self.mlp(x)
        x = self.ln2(x + mlp_out)

        return x

class DecisionTransformerGM(AgentDT):
    #def __init__(self, state_dim, act_dim, h_dim, h_dim, num_heads, num_blocks, max_timesteps, mlp_ratio, dropout, vocab_size, rtg_dim=1):
    def __init__(self, state_dim, act_dim, h_dim, num_heads, num_blocks, context_len, max_timesteps, mlp_ratio, dropout, rtg_dim=1):
        super().__init__()

        self.h_dim = h_dim   #Nº de Layers "nn.Linear" ~~ "h_dim"
        self.seq_len = 3*context_len

        # Construct embedding layer
        self.state_embed = nn.Linear(in_features=state_dim, out_features=h_dim)
        self.act_embed = nn.Linear(in_features=act_dim, out_features=h_dim)
        self.rtg_embed = nn.Linear(in_features=rtg_dim, out_features=h_dim)
        self.pos_embed = nn.Embedding(num_embeddings=max_timesteps, embedding_dim=h_dim)

        self.norm = nn.LayerNorm(h_dim)
        
        self.decoder_only = nn.Sequential(*([DecoderBlock(h_dim, num_heads, self.seq_len, mlp_ratio, dropout) for _ in range(num_blocks)]))  

        self.rtg_pred = nn.Linear(in_features=h_dim, out_features=1)
        self.state_pred = nn.Linear(in_features=h_dim, out_features=state_dim)
        self.act_pred = nn.Sequential(
            nn.Linear(h_dim, act_dim),
            nn.Tanh()
        )

    def forward(self, timestep, states, actions, returns_to_go):

        B, T, _ = states.shape # [batch size, seq length, h_dim]

        timestep_embedding = self.pos_embed(timestep)
        state_embedding = self.state_embed(states)
        act_embedding = self.act_embed(actions)
        rtg_embedding = self.rtg_embed(returns_to_go)

        state_embedding += timestep_embedding
        act_embedding += timestep_embedding
        rtg_embedding += timestep_embedding

        # (R{1}, S{1}, A{1}, ..., R{i}, S{i}, A{i}, ..., R{n}, S{n}, A{n}) | 1 < i < n
        stacked_inputs = torch.stack((rtg_embedding, state_embedding, act_embedding), dim=1) # [B, rtg_dim, state_dim, act_dim]
        stacked_inputs = stacked_inputs.permute(0, 2, 1, 3) #[B, state_dim, rtg_dim, act_dim]
        stacked_inputs = stacked_inputs.reshape(B, 3*T, self.h_dim) # [B, 3*T, hidden_size]  Nota: h_dim a.k.a "hidden_size"

        x = self.norm(stacked_inputs)
 
        x = self.decoder_only(x)

        x = x.reshape(B, T, 3, self.h_dim).permute(0, 2, 1, 3)  #[B, T, 3, hidden_size] --> [B, 3, T, hidden_size]     Nota: h_dim a.k.a "hidden_size"

        #returns_to_go_preds = self.rtg_pred(x[:,2])   # predict next return (t) given state (t-1) and action (t)    [0 state, 1 action, 2 rtg]
        #state_preds = self.state_pred(x[:,2])      # predict next state  (t) given state (t-1) and action (t)    [0, 1, 2 rtg]
        act_preds = self.act_pred(x[:,1])             # predict next action (t) given state (t-1)                   [0, 1, 2]
    
        return act_preds
class TrainableDTGM(DecisionTransformerGM):
    def __init__(self, state_dim, act_dim, h_dim, num_heads, num_blocks, context_len, max_timesteps, 
                 mlp_ratio, dropout, agent_num_button_actions, agent_num_camera_actions, agent_esc_button, rtg_dim=1):

        super().__init__(state_dim, act_dim, h_dim, num_heads, num_blocks, context_len, max_timesteps, mlp_ratio, dropout, rtg_dim)

        self.pos_embed = GlobalPositionEncoding(
            h_dim, max_timesteps + 1, context_len
        )

        self.state_dim = state_dim
        self.act_dim = act_dim
        
        self.agent_num_button_actions = agent_num_button_actions
        self.agent_num_camera_actions = agent_num_camera_actions
        self.agent_esc_button = agent_esc_button

        self.temperature_buttons = 1
        self.temperature_camera = 1
        self.temperature_esc = 1

        self.disable_esc_button = False

        configActionPolicy = ConfigActionPolicy(
            agent_num_button_actions=agent_num_button_actions,
            agent_num_camera_actions=agent_num_camera_actions,
            agent_esc_button=agent_esc_button,
            hidden_size=h_dim
        )

        self.act_pred = ActionPolicy(configActionPolicy)

    def forward(self, **kwargs):

        timestep = kwargs.get("timesteps")
        states = kwargs.get("states")
        action_targets = kwargs.get("actions")
        returns_to_go = kwargs.get("returns_to_go")
        attention_mask = kwargs.get("attention_mask")

        action_preds = super().forward(timestep, states, action_targets, returns_to_go)

        action_preds_button = action_preds["buttons"] * attention_mask.float().unsqueeze(-1)
        action_preds_camera = action_preds["camera"] * attention_mask.float().unsqueeze(-1)
        action_preds_esc = action_preds["esc"] * attention_mask.float().unsqueeze(-1)

        multi_categorical = MultiCategorical(action_preds_button, action_preds_camera, action_preds_esc)

        action_targets = action_targets * attention_mask.float().unsqueeze(-1)
        action_targets_button = action_targets[:, :, 0].long()
        action_targets_camera = action_targets[:, :, 1].long()
        action_targets_esc = action_targets[:, :, 2].long()
 
        log_probs = multi_categorical.log_prob(action_targets_button, action_targets_camera, action_targets_esc)

        loss = -log_probs.mean()

        return {"loss": loss}

    @torch.no_grad()
    def original_forward(self, **kwargs):

        timestep = kwargs.get("timesteps")
        states = kwargs.get("states")
        action_targets = kwargs.get("actions")
        returns_to_go = kwargs.get("returns_to_go")

        output = super().forward(timestep, states, action_targets, returns_to_go)
            
        return output

    def get_dt_action(self, states, actions, rewards, returns_to_go, timesteps, device, temperature_camera=None):

        states = states.reshape(1, -1, self.state_dim)
        actions = actions.reshape(1, -1, self.act_dim)
        returns_to_go = returns_to_go.reshape(1, -1, 1)
        timesteps = timesteps.reshape(1, -1)

        positions = self.seq_len

        states = states[:, -positions :]
        actions = actions[:, -positions :]
        returns_to_go = returns_to_go[:, -positions :]
        timesteps = timesteps[:, -positions :]
        padding = positions - states.shape[1]
        padding_actions = positions - actions.shape[1]

        # pad all tokens to sequence length
        attention_mask = torch.cat([torch.zeros(padding, device=device), torch.ones(states.shape[1], device=device)])
        attention_mask = attention_mask.to(dtype=torch.long).reshape(1, -1)
        states = torch.cat([torch.zeros((1, padding ,self.state_dim), device=device), states], dim=1).float()
        actions = torch.cat([torch.zeros((1, padding_actions, self.act_dim), device=device), actions], dim=1).float()
        returns_to_go = torch.cat([torch.zeros((1, padding, 1), device=device), returns_to_go], dim=1).float()

        timesteps = torch.cat([torch.zeros((1, padding), device=device, dtype=torch.long), timesteps], dim=1)

        action_logits = self.original_forward(
            states=states,
            actions=actions,
            rewards=rewards,
            returns_to_go=returns_to_go,
            timesteps=timesteps,
            attention_mask=attention_mask,
            return_dict=False,
        )

        action_logits_button = action_logits["buttons"][:,-1]
        action_logits_camera = action_logits["camera"][:,-1]
        action_logits_esc = action_logits["esc"][:,-1]

        multi_categorical = MultiCategorical(action_logits_button, action_logits_camera, action_logits_esc)

        if temperature_camera is None:
            temperature_camera = self.temperature_camera
        np_actions = multi_categorical.sample(self.temperature_buttons, temperature_camera, self.temperature_esc)

        actions = torch.tensor(np_actions, device=device, dtype=torch.long)

        action_preds_button = actions[0].unsqueeze(dim=0)
        action_preds_camera = actions[1].unsqueeze(dim=0)
        action_preds_esc = actions[2].unsqueeze(dim=0)

        if self.disable_esc_button:
            action_preds_esc = torch.zeros(1, device=device, dtype=torch.long)

        action_preds = torch.stack([action_preds_button, action_preds_camera, action_preds_esc], dim=-1)

        return action_preds
    


    def load_model(self, directory):

        if os.path.exists(directory):
            state_dict = safetensors.torch.load_file(os.path.join(directory, SAFE_WEIGHTS_NAME))
            self.load_state_dict(state_dict)
        else:
            raise FileNotFoundError(f"File not found {directory}.")
    
    @staticmethod
    def from_config(**config) -> AgentDT:
        agent_cfg = {
            "state_dim": config["embedding_dim"],
            "act_dim": AGENT_DT_ACTION_DIM,
            "h_dim": config["hidden_size"], 
            "num_heads": config["n_heads"],
            "num_blocks": config["n_layers"],
            "context_len": config["sequence_length"],
            "max_timesteps": config["max_ep_len"],
            "mlp_ratio": 1,
            "dropout": 0.1,
            "agent_num_button_actions":config["button_encoder_num_actions"],
            "agent_num_camera_actions":AGENT_DT_NUM_CAMERA_ACTIONS,
            "agent_esc_button":AGENT_DT_NUN_ESC_BUTTON
        }

        agent = TrainableDTGM(**agent_cfg)

        return agent
    
    @staticmethod
    def from_saved(**config)->'TrainableDTGM':

        agent_cfg = {
            "state_dim": config["embedding_dim"],
            "act_dim": AGENT_DT_ACTION_DIM,
            "h_dim": config["hidden_size"], 
            "num_heads": config["n_heads"],
            "num_blocks": config["n_layers"],
            "context_len": config["sequence_length"],
            "max_timesteps": config["max_ep_len"],
            "mlp_ratio": 1,
            "dropout": 0.1,
            "agent_num_button_actions":config["button_encoder_num_actions"],
            "agent_num_camera_actions":AGENT_DT_NUM_CAMERA_ACTIONS,
            "agent_esc_button":AGENT_DT_NUN_ESC_BUTTON
        }

        agent = TrainableDTGM(**agent_cfg)

        agent.load_model(config["models_dir"])

        return agent    