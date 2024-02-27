
from abc import abstractmethod
from collections import defaultdict
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import numpy as np
import os
import csv

from tqdm import tqdm
import math
import random
from dataclasses import dataclass


class MultiCategorical:
    def __init__(self, *logits):
        self._temperature = 1.0

        self.logits = logits
        self.dists = [Categorical(logits=logit) for logit in logits]

    def sample(self): 
        return [dist.sample() for dist in self.dists]

    def log_prob(self, *acts):

        lp = [dist.log_prob(act) for act, dist in zip(acts, self.dists)]

        #print(torch.stack(lp).mean(dim=1, keepdim=True).mean(dim=2))
        return sum(lp)

    def entropy(self):
        return sum(dist.entropy() for dist in self.dists)

    @property
    def greedy_action(self):
        return [torch.argmax(dist.logits, dim=-1) for dist in self.dists]
    
    def sample(self, *temperatures):

        samples = []

        for logit, temperature in zip(self.logits, temperatures):
            logit = logit.detach().cpu().numpy()
            logit = logit / temperature
            x = np.exp(logit - np.max(logit))
            probs = x / np.sum(x)
            probs = probs.flatten()
            sample = np.random.choice(probs.shape[0], p=probs)
            samples.append(sample)

        return samples

class ActEncoderDecoder:
    def __init__(self, button_act_csv_path, number_of_acts=256, embeddings_dir=None):
        self.number_of_acts = number_of_acts
        self.act_to_index = {}
        self.index_to_act = {}
        self.button_act_csv_path = button_act_csv_path

        if embeddings_dir and not os.path.exists(button_act_csv_path):
            self.generate_action_frequencies_csv(embeddings_dir)

        self.load_and_prepare()
    
    def load_and_prepare(self):
        with open(self.button_act_csv_path, mode='r') as csvfile:
            reader = csv.DictReader(csvfile)
            for i, row in enumerate(reader):
                if i >= self.number_of_acts:  
                    break
                act = row['Act']
                self.act_to_index[act] = i
                self.index_to_act[i] = act
    
    def encode(self, act):

        return self.act_to_index.get(str(act), None)
    
    def decode(self, index):
        return int(self.index_to_act.get(index, None))
    
    def generate_action_frequencies_csv(self, data_path):

        act_frequencies = defaultdict(int)

        episode_paths = self.get_all_npz_files_in_dir(data_path)

        for episode_path in tqdm(episode_paths, desc="Generating action frequencies csv"): 
                 
            data = np.load(episode_path)
            button_actions = data["button_actions"]
            
            for act in button_actions:
                act_frequencies[act] += 1

        sorted_acts = sorted(act_frequencies.items(), key=lambda item: item[1], reverse=True)
        
        unique_acts_count = len(sorted_acts)
        
        with open(self.button_act_csv_path, 'w', newline='') as csvfile:
            fieldnames = ['Act', 'Frequency']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for act, frequency in sorted_acts:
                writer.writerow({'Act': act, 'Frequency': frequency})
                
        return unique_acts_count, 'button_act_frequencies.csv'
            
    def get_all_npz_files_in_dir(self, dir_path):
        return glob.glob(os.path.join(dir_path, "*.npz"))


@dataclass
class DecisionTransformerGymEpisodeCollator:
    state_dim: int  # size of state space
    act_dim: int  # size of action space
    subset_training_len: int #subsets of the episode we use for training
    max_ep_len: int # max episode length in the dataset
    minibatch_samples: int # to store the number of trajectories in the dataset
    scale: float  # normalization of rewards/returns
    gamma: float # discount factor
    return_tensors: str = "pt"

    def __init__(self, state_dim: int, act_dim: int, subset_training_len: int, max_ep_len: int, minibatch_samples: int, gamma:float = 1.0, scale: float = 1 ) -> None:

        self.state_dim = state_dim
        self.act_dim = act_dim
        self.subset_training_len = subset_training_len
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

        s.append(np.array(feature["obs"][si : si + self.subset_training_len]).reshape(1, -1, self.state_dim))
        a.append(np.array(feature["acts"][si : si + self.subset_training_len]).reshape(1, -1, self.act_dim))
        r.append(np.array(feature["rewards"][si : si + self.subset_training_len]).reshape(1, -1, 1))

        d.append(np.array(feature["dones"][si : si + self.subset_training_len]).reshape(1, -1))

        timesteps.append(np.arange(si, si + self.subset_training_len).reshape(1, -1))

        timesteps[-1][timesteps[-1] >= self.max_ep_len] = self.max_ep_len - 1 
 
        rtg_sum = self._discount_cumsum(np.array(feature["rewards"]), gamma=self.gamma)[si: si + self.subset_training_len].reshape(1,-1,1)

        rtg.append(rtg_sum)

        if rtg[-1].shape[1] < s[-1].shape[1]:
            print("if true")
            rtg[-1] = np.concatenate([rtg[-1], np.zeros((1, 1, 1))], axis=1)

        tlen = s[-1].shape[1]
        s[-1] = np.concatenate([np.zeros((1, self.subset_training_len - tlen, self.state_dim)), s[-1]], axis=1)
        a[-1] = np.concatenate([np.zeros((1, self.subset_training_len - tlen, self.act_dim)),a[-1]], axis=1)
        r[-1] = np.concatenate([np.zeros((1, self.subset_training_len - tlen, 1)), r[-1]], axis=1)
        d[-1] = np.concatenate([np.ones((1, self.subset_training_len - tlen)) * 2, d[-1]], axis=1)
        rtg[-1] = np.concatenate([np.zeros((1, self.subset_training_len - tlen, 1)), rtg[-1]], axis=1) / self.scale

        mask.append(np.concatenate([np.zeros((1, self.subset_training_len - tlen)), np.ones((1, tlen))], axis=1))

    def __call__(self, features):

            
        s, a, r, d, rtg, timesteps, mask = [], [], [], [], [], [], []

        for feature in features:
            
            length = max(1,len(feature["rewards"]) - self.subset_training_len)
            population = list(range(length))
            weights = [math.sqrt(i) for i in range(1, length + 1)]

            for n in range(0, self.minibatch_samples):
                #si = random.randint(1, length)
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
    
class ConfigActionPolicy:
    def __init__(self, agent_num_button_actions, agent_num_camera_actions, agent_esc_button, hidden_size):
        self.agent_num_button_actions = agent_num_button_actions
        self.agent_num_camera_actions = agent_num_camera_actions
        self.agent_esc_button = agent_esc_button
        self.hidden_size = hidden_size

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

class AgentDT(nn.Module):
 

    def set_default_temperatures(self, temperature_buttons, temperature_camera, temperature_esc):
        self.temperature_buttons = temperature_buttons
        self.temperature_camera = temperature_camera
        self.temperature_esc = temperature_esc

    def set_disable_esc_button(self, disable_esc_button=True):
        self.disable_esc_button = disable_esc_button

    @staticmethod
    @abstractmethod
    def from_config(**config)->'AgentDT':
        pass

    @staticmethod
    @abstractmethod
    def from_saved(**config)->'AgentDT':
        pass


        
