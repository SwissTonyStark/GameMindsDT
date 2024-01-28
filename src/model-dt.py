import torch.nn as nn
class DecisionTransformer(nn.Module):
    def __init__(self, modelT, ffn_dim, state_dim, act_dim, max_timesteps, rtg_dim=1):
        
        self.hidden_size = ffn_dim
        # Construct embedding layer
        self.state_embed = nn.Linear(in_features=state_dim, out_features=ffn_dim)
        self.act_embed = nn.Linear(in_features=act_dim, out_features=ffn_dim)
        self.rtg_embed = nn.Linear(in_features=1, out_features=ffn_dim)
        self.pos_embed = nn.Embedding(num_embeddings=max_timesteps, embedding_dim=ffn_dim)

        self.norm = nn.LayerNorm(ffn_dim)

        #TODO: Complete Basic Transformer parameters
        self.transformerGPT = modelT(config)

        self.rtg_pred = nn.Linear(in_features=ffn_dim, out_features=1)
        self.state_pred = nn.Linear(in_features=ffn_dim, out_features=state_dim)
        self.act_pred = nn.Sequential(
            nn.Linear(hidden_size, act_dim),
            nn.Tanh()
        )
    
    def forward(self, timestep, max_timesteps, states, actions, returns_to_go):

        batch_size, tokens, _ = states.shape

        pos_embedding = self.pos_embed(num_timesteps, timestep)
        
        state_embedding = self.state_embed(states)
        act_embedding = self.act_embed(actions)
        rtg_embedding = self.rtg_embed(returns_to_go)

        state_embedding += pos_embedding
        act_embedding += pos_embedding
        returns_to_go += pos_embedding

        stacked_inputs = torch.stack((rtg_embedding, state_embedding, act_embedding), dim=1)
        stacked_inputs = stacked_inputs.permute(0, 2, 1, 3)
        stacked_inputs = stacked_inputs.reshape(batch_size, 3*tokens, self.hidden_size)
        
        inp = self.norm(stacked_inputs)
        
        #TODO: Complete Basic Transformer
        out = self.transformerGPT(inp)
        out = out.reshape(batch_size, tokens, 3, self.hidden_size).permute(0, 2, 1, 3)

        returns_to_go_preds = self.rtg_pred(x[:,2])     # predict next return given state and action
        state_preds = self.state_pred(x[:,2])           # predict next state given state and action
        act_preds = self.act_pred(x[:,1])               # predict next action given state
        
        return returns_to_go, state_preds, act_preds

def train():
    return
