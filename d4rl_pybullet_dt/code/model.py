import torch
import torch.nn as nn

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

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

        last_tt = last_t.type(torch.int64)
        # (B, Tmax, N) -> (B, 1, N)
        global_embedding = torch.gather(batched_global_embedding, 1, last_tt)
        # (1, 3 * Cmax, N) -> (1, T, N)
        block_embedding = self._block_position_embedding()[:, :context_size, :]
        # (B, 1, N) + (1, T, N) -> (B, T, N)

        return global_embedding + block_embedding
    

class DecoderBlock(nn.Module):
    def __init__(self, h_dim, num_heads, seq_len, mlp_ratio, dropout):
        super().__init__()
        self.attn = MaskedSelfAttention(h_dim, num_heads, seq_len, dropout)
        self.ln1 = nn.LayerNorm(h_dim)
        self.mlp = MLP(h_dim, mlp_ratio, dropout)
        self.ln2 = nn.LayerNorm(h_dim)
    def forward(self, x):

        attn_out= self.attn(x)# add residual
        x = self.ln1(x + attn_out)
        mlp_out = self.mlp(x)
        x = self.ln2(x + mlp_out)
        
        return x
    

class DecisionTransformer(nn.Module):
    def __init__(self, state_dim, act_dim, h_dim, num_heads, num_blocks, context_len, max_timesteps, mlp_ratio, dropout, rtg_dim=1):
        super().__init__()

        self.h_dim = h_dim   #Nº de Layers "nn.Linear" ~~ "h_dim"
        self.seq_len = 3*context_len

        # Construct embedding layer
        self.state_embed = nn.Linear(in_features=state_dim, out_features=h_dim)
        self.act_embed = nn.Linear(in_features=act_dim, out_features=h_dim)
        self.rtg_embed = nn.Linear(in_features=rtg_dim, out_features=h_dim)
        #self.pos_embed = nn.Embedding(num_embeddings=max_timesteps, embedding_dim=h_dim)
        self.pos_embed = GlobalPositionEncoding(h_dim, max_timesteps + 1, context_len)

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

        returns_to_go_preds = self.rtg_pred(x[:,2])   # predict next return (t) given state (t-1) and action (t)    [0 state, 1 action, 2 rtg]
        state_preds = self.state_pred(x[:,2])      # predict next state  (t) given state (t-1) and action (t)    [0, 1, 2 rtg]
        act_preds = self.act_pred(x[:,1])             # predict next action (t) given state (t-1)                   [0, 1, 2]
    
        return returns_to_go_preds, state_preds, act_preds