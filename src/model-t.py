import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import math
import numpy as np
import seaborn as sns
import matplotlib.pylab as plt

# ATTENTION MULTI MASKED 
class MaskedSelfAttention(nn.Module):

    def __init__(self, embed_dim, num_heads, seq_len, dropout):
        super().__init__()

        self.embed_dim = embed_dim # embeding dimensionality, includes all heads
        self.num_heads = num_heads #  num heads
        assert self.n_embd % self.n_head == 0 , \
            "Embedding dimension must be multiple of the number of heads."
        
        self.seq_len = seq_len

        # key, query, value projections 
        self.proj_q = nn.Linear(embed_dim, embed_dim)
        self.proj_k = nn.Linear(embed_dim, embed_dim)
        self.proj_v = nn.Linear(embed_dim, embed_dim)
        self.proj_o = nn.Linear(embed_dim, embed_dim)

        # output projection
        self.proj_out = nn.Linear(self.n_embd, self.n_embd)
        # regularization
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape # batch size, sequence length, embedding dimensionality (n_embd)
        head_size = self.n_head, C // self.n_head

        # calculate query, key, values
        q = self.proj_q(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, seqLen, numHeads, headSize) -> (B, numHeads, seqLen, headSize)
        k = self.proj_k(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, seqLen, numHeads, headSize) -> (B, numHeads, seqLen, headSize)
        v = self.proj_v(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, seqLen, numHeads, headSize) -> (B, numHeads, seqLen, headSize)

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
        y = self.resid_dropout(out)
        return y

# MLP
class MLP(nn.Module):

    def __init__(self, embed_dim, ffn_dim, dropout):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, ffn_dim)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(ffn_dim, embed_dim)
        self.drop= nn.Dropout(dropout)

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.drop(self.fc2(x))
        return x

# BLOCK

class DecoderBlock(nn.Module):

    def __init__(self, embed_dim, num_heads, seq_len, dropout, ffn_ratio=4):
        super().__init__()

        self.attn = MaskedSelfAttention(embed_dim, num_heads, seq_len, dropout)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, int(embed_dim * ffn_ratio))
        self.ln_2 = nn.LayerNorm(self.embed_dim)

    def forward(self, x):
        x = self.attn(x) + x # add residual
        x = self.ln1(x) # normalize
        x = self.mlp(x) + x
        out = self.ln2(x)
        return out

# MAIN TRANSFORMER

# Stack of Decoder Blocks
self.blocks = nn.ModuleList([
    DecoderBlock(embed_size, heads, mlp_ratio)
    for _ in range(num_blocks)
])


    