import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader,TensorDataset

import math
import numpy as np
#import seaborn as sns
import matplotlib.pylab as plt

#Especifico para el gym+dataset "D4RL_Pybullet"
import gym
import d4rl_pybullet


class MaskedSelfAttention(nn.Module):

    def __init__(self, embed_dim, num_heads, seq_len, dropout):
        super().__init__()

        self.embed_dim = embed_dim # embeding dimensionality, includes all heads
        self.num_heads = num_heads #  num heads
        assert self.embed_dim % self.num_heads == 0 , \
            "Embedding dimension must be multiple of the number of heads."

        self.seq_len = seq_len

        # key, query, value projections
        self.proj_q = nn.Linear(embed_dim, embed_dim)
        self.proj_k = nn.Linear(embed_dim, embed_dim)
        self.proj_v = nn.Linear(embed_dim, embed_dim)

        # output projection
        self.proj_out = nn.Linear(self.embed_dim, self.embed_dim)
        # regularization
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x):
        print("T FORWARD (tocho)")
        B, T, C = x.shape # batch size, sequence length, embedding dimensionality (embed_dim)
        #head_size = self.num_heads, C // self.num_heads

        # calculate query, key, values
        q = self.proj_q(x).view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2) # (B, seqLen, numHeads, headSize) -> (B, numHeads, seqLen, headSize)
        k = self.proj_k(x).view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2) # (B, seqLen, numHeads, headSize) -> (B, numHeads, seqLen, headSize)
        v = self.proj_v(x).view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2) # (B, seqLen, numHeads, headSize) -> (B, numHeads, seqLen, headSize)

        # causal self-attention; Self-attend: (B, numHeads, seqLen, headSize) x (B, numHeads, headSize, seqLen) -> (B, numHeads, seqLen, seqLen)
        # scaled_dot_product
        attn_logits = (q @ k.transpose(-2, -1))
        attn_logits = attn_logits / torch.sqrt(torch.tensor(k.size(-1)))
        # apply mask

        mask = torch.zeros(x.shape[1], x.shape[0]).bool() #toDevice
        subsequent_mask = torch.triu(torch.ones(B, T, T), 1).bool() #toDevice
        selfattn_mask = subsequent_mask + mask.unsqueeze(-2)
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

    def __init__(self, embed_dim, ffn_dim, dropout):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, ffn_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(ffn_dim, embed_dim)
        self.drop= nn.Dropout(dropout)

    def forward(self, x):
        print("MLP FORWARD")
        x = self.act(self.fc1(x))
        x = self.drop(self.fc2(x))
        return x

class DecoderBlock(nn.Module):

    def __init__(self, embed_dim, num_heads, seq_len, mlp_ratio, dropout):
        super().__init__()

        self.attn = MaskedSelfAttention(embed_dim, num_heads, seq_len, dropout)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, int(embed_dim * mlp_ratio),dropout)
        self.ln_2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        print("DECODER BLOCK FORWARD")
        x = self.ln1(x) # normalize
        x = self.attn(x) + x # add residual
        x = self.ln2(x)
        x = self.mlp(x) + x

        return x

class DecisionTransformer(nn.Module):
    def __init__(self, state_dim, act_dim, ffn_dim, embed_dim, num_heads, num_blocks, max_timesteps, mlp_ratio, dropout, vocab_size, rtg_dim=1):
        super().__init__()

        self.ffn_dim = ffn_dim   #Nº de Layers "nn.Linear" ~~ "ffn_dim"
        self.seq_len = act_dim   # Omar/Shuang-> Provisional, revisar esta linea.
        # Construct embedding layer
        self.state_embed = nn.Linear(in_features=state_dim, out_features=ffn_dim)
        self.act_embed = nn.Linear(in_features=act_dim, out_features=ffn_dim)
        self.rtg_embed = nn.Linear(in_features=rtg_dim, out_features=ffn_dim)
        self.pos_embed = nn.Embedding(num_embeddings=max_timesteps, embedding_dim=ffn_dim)

        self.norm = nn.LayerNorm(ffn_dim)

        #TODO: Complete Basic Transformer parameters
        self.transformerGPT = nn.ModuleList([DecoderBlock(embed_dim, num_heads, self.seq_len, mlp_ratio, dropout) for _ in range(num_blocks)])

        self.rtg_pred = nn.Linear(in_features=ffn_dim, out_features=1)
        self.state_pred = nn.Linear(in_features=ffn_dim, out_features=state_dim)
        self.act_pred = nn.Sequential(
            nn.Linear(ffn_dim, act_dim),
            nn.Tanh()
        )

    def forward(self, timestep, max_timesteps, states, actions, returns_to_go):
        print("DT FORWARD")
        print(timestep)
        print(max_timesteps)
        print(states)
        print(actions)
        print(returns_to_go)
        B, T, _ = states.shape # [batch size, seq length, embed_dim]

        pos_embedding = self.pos_embed(max_timesteps, timestep)
        print("DEBUG 1")
        state_embedding = self.state_embed(states)
        act_embedding = self.act_embed(actions)
        rtg_embedding = self.rtg_embed(returns_to_go)
        print("DEBUG 2")
        state_embedding += pos_embedding
        act_embedding += pos_embedding
        returns_to_go += pos_embedding
        print("DEBUG 3")
        # (R{1}, S{1}, A{1}, ..., R{i}, S{i}, A{i}, ..., R{n}, S{n}, A{n}) | 1 < i < n
        stacked_inputs = torch.stack((rtg_embedding, state_embedding, act_embedding), dim=1) # [B, rtg_dim, state_dim, act_dim]
        stacked_inputs = stacked_inputs.permute(0, 2, 1, 3) #[B, state_dim, rtg_dim, act_dim]
        stacked_inputs = stacked_inputs.reshape(B, 3*T, self.ffn_dim) # [B, 3*T, hidden_size]  Nota: ffn_dim a.k.a "hidden_size"
        print("DEBUG 4")
        x = self.norm(stacked_inputs)
        if torch.is_tensor(x):
            print("SHAP INPUT TRANSFORMER TOCHO: ", x.shape)
        else:
            print("LEN INPUT T: ", len(x))
        print(x)
        #TODO: Complete Basic Transformer
        out = self.transformerGPT(x)
        out = out.reshape(B, T, 3, self.ffn_dim).permute(0, 2, 1, 3)  #[B, T, 3, hidden_size] --> [B, 3, T, hidden_size]     Nota: ffn_dim a.k.a "hidden_size"

        returns_to_go_preds = self.rtg_pred(out[:,2])     # predict next return given state and action [0 state, 1 action, 2 rtg]
        state_preds = self.state_pred(out[:,2])           # predict next state given state and action  [0, 1, 2 rtg]
        act_preds = self.act_pred(out[:,1])               # predict next action given state            [0, 1, 2]

        return returns_to_go, state_preds, act_preds