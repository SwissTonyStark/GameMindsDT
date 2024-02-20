import math
from transformers import DecisionTransformerModel
import torch


import numpy as np
import random
from dataclasses import dataclass
from torch import nn

from .dt_model_common import MultiCategorical

@dataclass
class DecisionTransformerGymEpisodeCollator:
    state_dim: int  # size of state space
    act_dim: int  # size of action space
    sequence_length: int #subsets of the episode we use for training
    max_ep_len: int # max episode length in the dataset
    minibatch_samples: int # to store the number of trajectories in the dataset
    scale: float  # normalization of rewards/returns
    gamma: float # discount factor
    return_tensors: str = "pt"

    def __init__(self, state_dim: int, act_dim: int, sequence_length: int, max_ep_len: int, minibatch_samples: int, gamma:float = 1.0, scale: float = 1 ) -> None:

        self.state_dim = state_dim
        self.act_dim = act_dim
        self.sequence_length = sequence_length
        self.max_ep_len = max_ep_len
        self.minibatch_samples = minibatch_samples
        self.gamma = gamma
        self.scale = scale


    def _discount_cumsum(self, x, gamma):
        discount_cumsum = np.zeros_like(x)
        discount_cumsum[-1] = x[-1]
        for t in reversed(range(x.shape[0] - 1)):
            discount_cumsum[t] = x[t] + gamma * discount_cumsum[t + 1]
        return np.array(list(reversed(discount_cumsum)))

    def sample(self, feature, si, s, a, r, d, rtg, timesteps, mask):

        s.append(np.array(feature["obs"][si : si + self.sequence_length]).reshape(1, -1, self.state_dim))
        a.append(np.array(feature["acts"][si : si + self.sequence_length]).reshape(1, -1, self.act_dim))
        r.append(np.array(feature["rewards"][si : si + self.sequence_length]).reshape(1, -1, 1))

        d.append(np.array(feature["dones"][si : si + self.sequence_length]).reshape(1, -1))

        timesteps.append(np.arange(si, si + self.sequence_length).reshape(1, -1))

        timesteps[-1][timesteps[-1] >= self.max_ep_len] = self.max_ep_len - 1 
 
        rtg_sum = self._discount_cumsum(np.array(feature["rewards"]), gamma=self.gamma)[si: si + self.sequence_length].reshape(1,-1,1)

        rtg.append(rtg_sum)

        if rtg[-1].shape[1] < s[-1].shape[1]:
            print("if true")
            rtg[-1] = np.concatenate([rtg[-1], np.zeros((1, 1, 1))], axis=1)

        tlen = s[-1].shape[1]
        s[-1] = np.concatenate([np.zeros((1, self.sequence_length - tlen, self.state_dim)), s[-1]], axis=1)
        a[-1] = np.concatenate(
            [np.zeros((1, self.sequence_length - tlen, self.act_dim)), a[-1]],
            axis=1,
        )
        r[-1] = np.concatenate([np.zeros((1, self.sequence_length - tlen, 1)), r[-1]], axis=1)
        d[-1] = np.concatenate([np.ones((1, self.sequence_length - tlen)) * 2, d[-1]], axis=1)
        rtg[-1] = np.concatenate([np.zeros((1, self.sequence_length - tlen, 1)), rtg[-1]], axis=1) / self.scale

        mask.append(np.concatenate([np.zeros((1, self.sequence_length - tlen)), np.ones((1, tlen))], axis=1))

    def __call__(self, features):

        #if (False):
        batch_size = len(features)

        traj_lens = []
        for feature in features:
            obs = feature["obs"]
            traj_lens.append(len(obs))
            

        traj_lens = np.array(traj_lens)
        p_sample = traj_lens / sum(traj_lens)

        batch_inds = np.random.choice(
            np.arange(batch_size),
            size=self.minibatch_samples,
            replace=True,
            p=p_sample,  
        )

        s, a, r, d, rtg, timesteps, mask = [], [], [], [], [], [], []

        for feature in features:
            
            length = max(1,len(feature["rewards"]) - self.sequence_length)
            population = list(range(length))

            weights = [math.sqrt(i) for i in range(1, length + 1)]

            for n in range(0, self.minibatch_samples):
                #si = random.randint(0, len(feature["rewards"]) - 1)
                si = random.choices(population, weights=weights, k=1)[0]
                self.sample(feature, si, s, a, r, d, rtg, timesteps, mask)


        s = torch.from_numpy(np.concatenate(s, axis=0)).float()
        a = torch.from_numpy(np.concatenate(a, axis=0)).float()
        r = torch.from_numpy(np.concatenate(r, axis=0)).float()
        d = torch.from_numpy(np.concatenate(d, axis=0))
        rtg = torch.from_numpy(np.concatenate(rtg, axis=0)).float()
        timesteps = torch.from_numpy(np.concatenate(timesteps, axis=0)).long()
        mask = torch.from_numpy(np.concatenate(mask, axis=0)).float()

        return {
            "states": s,
            "actions": a,
            "rewards": r,
            "returns_to_go": rtg,
            "timesteps": timesteps,
            "attention_mask": mask,
            "return_loss": True,
        }


class ActionPolicy(nn.Module):
    def __init__(self, config):
        super(ActionPolicy, self).__init__()
        self.agent_num_button_actions = config.agent_num_button_actions
        self.agent_num_camera_actions = config.agent_num_camera_actions
        self.agent_esc_button = config.agent_esc_button

        self.predict_action = nn.ModuleDict({
            'buttons': nn.Sequential(
                nn.Linear(config.hidden_size, self.agent_num_button_actions),
            ),
            'camera': nn.Sequential(
                nn.Linear(config.hidden_size, self.agent_num_camera_actions),
            ),
            'esc': nn.Sequential(
                nn.Linear(config.hidden_size, self.agent_esc_button),
            )
        })

    def forward(self, x):
        actions = {action_type: module(x) for action_type, module in self.predict_action.items()}
        return actions
    
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

class TrainableDT(DecisionTransformerModel):
    def __init__(self, config):
        super().__init__(config)

        self.embed_timestep = GlobalPositionEncoding(
            config.hidden_size, config.max_ep_len + 1, config.n_positions // 3
        )

        self.state_dim = config.state_dim
        self.act_dim = config.act_dim
        
        self.agent_num_button_actions = config.agent_num_button_actions
        self.agent_num_camera_actions = config.agent_num_camera_actions
        self.agent_esc_button = config.agent_esc_button

        self.temperature_buttons = 1
        self.temperature_camera = 1
        self.temperature_esc = 1

        self.disable_esc_button = False

        self.predict_action = ActionPolicy(config)

    def set_default_temperatures(self, temperature_buttons, temperature_camera, temperature_esc):
        self.temperature_buttons = temperature_buttons
        self.temperature_camera = temperature_camera
        self.temperature_esc = temperature_esc

    def set_disable_esc_button(self, disable_esc_button=True):
        self.disable_esc_button = disable_esc_button

    def forward(self, **kwargs):
        del kwargs["return_loss"]
        output = super().forward(**kwargs)
        action_targets = kwargs.get("actions")
        attention_mask = kwargs.get("attention_mask")

        if 'return_dict' in kwargs:
            action_preds = output.action_preds
        else:
            action_preds = output[1]

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

    def original_forward(self, **kwargs):

        output = super().forward(**kwargs)
            
        return output

    def get_dt_action(self, states, actions, rewards, returns_to_go, timesteps, device, temperature_camera=None):

        states = states.reshape(1, -1, self.state_dim)
        actions = actions.reshape(1, -1, self.act_dim)
        returns_to_go = returns_to_go.reshape(1, -1, 1)
        timesteps = timesteps.reshape(1, -1)

        positions = self.config.n_positions // 3

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

        state_preds, action_logits, return_preds = self.original_forward(
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
