import torch.nn as nn
from model_T import DecoderBlock

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

        B, T, _ = states.shape # [batch size, seq length, embed_dim]

        pos_embedding = self.pos_embed(max_timesteps, timestep)

        state_embedding = self.state_embed(states)
        act_embedding = self.act_embed(actions)
        rtg_embedding = self.rtg_embed(returns_to_go)

        state_embedding += pos_embedding
        act_embedding += pos_embedding
        returns_to_go += pos_embedding

        # (R{1}, S{1}, A{1}, ..., R{i}, S{i}, A{i}, ..., R{n}, S{n}, A{n}) | 1 < i < n
        stacked_inputs = torch.stack((rtg_embedding, state_embedding, act_embedding), dim=1) # [B, rtg_dim, state_dim, act_dim]
        stacked_inputs = stacked_inputs.permute(0, 2, 1, 3) #[B, state_dim, rtg_dim, act_dim]
        stacked_inputs = stacked_inputs.reshape(B, 3*T, self.ffn_dim) # [B, 3*T, hidden_size]  Nota: ffn_dim a.k.a "hidden_size"

        x = self.norm(stacked_inputs)

        #TODO: Complete Basic Transformer
        out = self.transformerGPT(x)
        out = out.reshape(B, T, 3, self.ffn_dim).permute(0, 2, 1, 3)  #[B, T, 3, hidden_size] --> [B, 3, T, hidden_size]     Nota: ffn_dim a.k.a "hidden_size"

        returns_to_go_preds = self.rtg_pred(out[:,2])     # predict next return given state and action [0 state, 1 action, 2 rtg]
        state_preds = self.state_pred(out[:,2])           # predict next state given state and action  [0, 1, 2 rtg]
        act_preds = self.act_pred(out[:,1])               # predict next action given state            [0, 1, 2]

        return returns_to_go, state_preds, act_preds