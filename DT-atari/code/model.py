import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

# alternative position embedding
# Thanks to d3lply library for the following code Parameter and GlobalPositionEncoding

class Parameter(nn.Module):  # type: ignore
    _parameter: nn.Parameter
    def __init__(self, data: torch.Tensor):
        super().__init__()
        self._parameter = nn.Parameter(data)
    def forward(self) -> torch.Tensor:
        return self._parameter
    def __call__(self) -> torch.Tensor:
        return super().__call__()
    @property
    def data(self) -> torch.Tensor:
        return self._parameter.data

class GlobalPositionEncoding(nn.Module):
    def __init__(self, embed_dim: int, max_timestep: int, context_size: int):
        super().__init__()
        self._embed_dim = embed_dim
        self._global_position_embedding = Parameter(
            torch.zeros(1, max_timestep, embed_dim, dtype=torch.float32)
        )
        self._block_position_embedding = Parameter(
            torch.zeros(1, 3 * context_size, embed_dim, dtype=torch.float32)
        )
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        assert t.dim() == 2, "Expects (B, T)"
        batch_size, context_size = t.shape
        # (B, 1, 1) -> (B, 1, N)
        last_t = torch.repeat_interleave(
            t[:, -1].view(-1, 1, 1), self._embed_dim, dim=-1
        )
        # (1, Tmax, N) -> (B, Tmax, N)
        batched_global_embedding = torch.repeat_interleave(
            self._global_position_embedding(),
            batch_size,
            dim=0,
        )
        # (B, Tmax, N) -> (B, 1, N)
        global_embedding = torch.gather(batched_global_embedding, 1, last_t)
        # (1, 3 * Cmax, N) -> (1, T, N)
        block_embedding = self._block_position_embedding()[:, :context_size, :]
        # (B, 1, N) + (1, T, N) -> (B, T, N)
        return global_embedding + block_embedding


class MaskedSelfAttention(nn.Module):
    def __init__(self, h_dim, seq_len, n_heads, drop_p):
        super().__init__()

        self.h_dim = h_dim # embeding dimensionality, includes all heads
        self.n_heads = n_heads #  num heads
        self.seq_len = seq_len # sequence length
        
        assert self.h_dim % self.n_heads == 0 , \
            "Embedding dimension must be multiple of the number of heads."

        # key, query, value projections for all heads, but in a batch
        self.proj_q = nn.Linear(self.h_dim, self.h_dim)
        self.proj_k = nn.Linear(self.h_dim, self.h_dim)
        self.proj_v = nn.Linear(self.h_dim, self.h_dim)

        # output projection
        self.proj_out = nn.Linear(self.h_dim, self.h_dim)

        # regularization
        self.att_drop = nn.Dropout(drop_p)
        self.proj_drop = nn.Dropout(drop_p)

        # causal mask to ensure that attention is only applied to the left in the input sequence
        # every token only comunicates with the previous ones
        ones = torch.ones((seq_len, seq_len))
        mask = torch.tril(ones).view(1, 1, seq_len, seq_len)

        # register buffer makes sure mask does not get updated
        # during backpropagation
        self.register_buffer('mask',mask)

    def forward(self, x):
        B, T, C = x.shape # batch size, seq length, h_dim * n_heads

        N, D = self.n_heads, C // self.n_heads # N = num heads, D = attention dim

        # rearrange q, k, v as (B, N, T, D)
        q = self.proj_q(x).view(B, T, N, D).transpose(1,2)  # (B, seqLen, numHeads, headSize) -> (B, numHeads, seqLen, headSize)
        k = self.proj_k(x).view(B, T, N, D).transpose(1,2)
        v = self.proj_v(x).view(B, T, N, D).transpose(1,2)

        # causal self-attention; Self-attend: (B, numHeads, seqLen, headSize) x (B, numHeads, headSize, seqLen) -> (B, numHeads, seqLen, seqLen)
        # scaled_dot_product
        attn_logits = q @ k.transpose(2,3) / math.sqrt(D)
        # causal mask applied to attn_logits
        attn_logits = attn_logits.masked_fill(self.mask[...,:T,:T] == 0, float('-inf'))
        # normalize attn_logits, all -inf -> 0 after softmax
        attention = F.softmax(attn_logits, dim=-1)

        # attention (B, N, T, D)
        attention = self.att_drop(attention @ v) # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

        # gather heads and project (B, N, T, D) -> (B, T, N*D)
        attention = attention.transpose(1, 2).contiguous().view(B,T,N*D) # re-assemble all head outputs side by side

        out = self.proj_drop(self.proj_out(attention))
        return out


class DecoderBlock(nn.Module):
    def __init__(self, h_dim, seq_len, n_heads, drop_p):
        super().__init__()
        # self attention
        self.attention = MaskedSelfAttention(h_dim, seq_len, n_heads, drop_p)
      
        self.mlp = nn.Sequential(
                nn.Linear(h_dim, 4*h_dim),  # mlp_ratio=4
                nn.GELU(),
                nn.Linear(4*h_dim, h_dim),  # mlp_ratio=4
                nn.Dropout(drop_p),
            )
        # Layer normalization
        self.ln1 = nn.LayerNorm(h_dim)
        self.ln2 = nn.LayerNorm(h_dim)

    def forward(self, x):
        # Attention -> LayerNorm -> MLP -> LayerNorm
        x = x + self.attention(x) # residual connection
        x = self.ln1(x)
        x = x + self.mlp(x) # residual connection
        x = self.ln2(x)
        return x


class DecisionTransformer(nn.Module):
    def __init__(self, state_dim, act_voc, n_blocks, h_dim, context_len,
                 n_heads, drop_p, max_timestep=4096):
        super().__init__()

        self.state_dim = state_dim
        self.act_voc = act_voc # action vocabulary
        self.h_dim = h_dim # n_embd
        self.n_blocks = n_blocks # num decoder blocks
        self.context_len = context_len
        self.n_heads = n_heads
        self.drop_p = drop_p

        # TIMESTEP embedding: the positional embeddings are learned.
        #self.embed_timestep = nn.Embedding(max_timestep, self.h_dim)
        self.embed_timestep = GlobalPositionEncoding(self.h_dim, max_timestep + 1, self.context_len)
        # RTG embedding
        self.embed_rtg = torch.nn.Linear(1, self.h_dim)

        # STATE embedding
        # encoder channels 32, 64, 64
        # encoder filter sizes 8x8, 4x4, 3x3
        # encoder strides 4,2,1
        self.embed_state = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4, padding=0), # stack 4 frames -> 4 channel in
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, self.h_dim),
            nn.Tanh())

        # discrete ACTION embedding
        self.embed_action = torch.nn.Embedding(self.act_voc, self.h_dim)

        # layer normalization
        self.norm = nn.LayerNorm(h_dim)

        # TRANSFORMER decoder blocks
        input_seq_len = 3 * self.context_len 
        blocks = [DecoderBlock(self.h_dim, input_seq_len, self.n_heads, self.drop_p) for _ in range(self.n_blocks)]
        self.transformer = nn.Sequential(*blocks)

        # output layer / action prediction
        self.predict_action = nn.Linear(self.h_dim, self.act_voc) # no nn.Tanh() for discrete actions


    def forward(self, timesteps, states, actions, returns_to_go):
        # states: (batch, seq_len, 4*84*84) stacked images
        # actions: (batch, seq_len, 1)
        # targets: (batch, seq_len, 1)
        # rtgs: (batch, seq_len, 1)
        # timesteps: (batch, 1, 1)

        B, T, _ = states.shape # [batch size, seq_len, h_dim]

        # PLAN (original paper):
        # 1. Compute embeddings for tokens
        # pos_embedding = embed_t(t) # per-timestep (note: not per-token)
        # s_embedding = embed_s(s) + pos_embedding
        # a_embedding = embed_s(a) + pos_embedding
        # R_embedding = embed_R(R) + pos_embedding

        # time embeddings as positional embeddings
        time_embeddings = self.embed_timestep(timesteps)
        # flatten to pass though the conv as an array of 4channel images
        states = states.reshape(-1, 4, 84, 84) # 4 stacked images (84,84) hardcoded 
        # state embedding
        state_embeddings = self.embed_state(states).reshape(B, T, self.h_dim) # ->(batch, context_length, n_embd)
        state_embeddings = state_embeddings + time_embeddings
        action_embeddings = self.embed_action(actions.type(torch.long).squeeze(-1)) + time_embeddings
        returns_embeddings = self.embed_rtg(returns_to_go) + time_embeddings

        # interleave tokens as (R_1, s_1, a_1, ..., R_K, s_K)
        h = torch.stack(
            (returns_embeddings, state_embeddings, action_embeddings), dim=1
        ).permute(0, 2, 1, 3).reshape(B, 3 * T, self.h_dim)

        # layer norm
        h = self.norm(h)

        # use transformer to get hidden states
        h = self.transformer(h)

        h = h.reshape(B, T, 3, self.h_dim).permute(0, 2, 1, 3)  #[B, T, 3, h_dim] --> [B, 3, T, h_dim] 

        # action predictions
        action_preds = self.predict_action(h[:,1])  # predict action given r, s

        return action_preds
