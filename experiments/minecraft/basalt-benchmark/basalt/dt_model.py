import math
from transformers import DecisionTransformerConfig, DecisionTransformerModel
import torch

import numpy as np
import random
from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader
from torch import nn
import torch.nn.functional as F
from typing import Optional

from basalt.vpt_lib.agent import AGENT_NUM_BUTTON_ACTIONS, AGENT_NUM_CAMERA_ACTIONS

@dataclass
class DecisionTransformerGymEpisodeCollator:
    return_tensors: str = "pt"
    max_len: int = 64 #subsets of the episode we use for training
    state_dim: int = 17  # size of state space
    act_dim: int = 6  # size of action space
    max_ep_len: int = 10000 # max episode length in the dataset
    scale: float = 1  # normalization of rewards/returns
    p_sample: np.array = None  # a distribution to take account trajectory lengths
    minibatch_samples: int = 16 # to store the number of trajectories in the dataset

    def __init__(self, state_dim=None, act_dim=None) -> None:

        if state_dim is not None:
            self.state_dim = state_dim
        if act_dim is not None:
            self.act_dim = act_dim


    def _discount_cumsum(self, x, gamma):
        discount_cumsum = np.zeros_like(x)
        discount_cumsum[-1] = x[-1]
        for t in reversed(range(x.shape[0] - 1)):
            discount_cumsum[t] = x[t] + gamma * discount_cumsum[t + 1]
        return np.array(list(reversed(discount_cumsum)))

    def sample(self, feature, si, s, a, r, d, rtg, timesteps, mask):

        # get sequences from dataset
        s.append(np.array(feature["obs"][si : si + self.max_len]).reshape(1, -1, self.state_dim))
        a.append(np.array(feature["acts"][si : si + self.max_len]).reshape(1, -1, self.act_dim))
        r.append(np.array(feature["rewards"][si : si + self.max_len]).reshape(1, -1, 1))

        d.append(np.array(feature["dones"][si : si + self.max_len]).reshape(1, -1))

        #si_time = random.randint(0, self.max_ep_len - self.max_len - 1)

        timesteps.append(np.arange(si, si + self.max_len).reshape(1, -1))
        timesteps[-1][timesteps[-1] >= self.max_ep_len] = self.max_ep_len - 1  # padding cutoff
 
        rtg_sum = self._discount_cumsum(np.array(feature["rewards"]), gamma=0.99)[si: si + self.max_len].reshape(1,-1,1)

        rtg.append(rtg_sum)

        if rtg[-1].shape[1] < s[-1].shape[1]:
            print("if true")
            rtg[-1] = np.concatenate([rtg[-1], np.zeros((1, 1, 1))], axis=1)

        # padding and state + reward normalization
        tlen = s[-1].shape[1]
        s[-1] = np.concatenate([np.zeros((1, self.max_len - tlen, self.state_dim)), s[-1]], axis=1)
        a[-1] = np.concatenate(
            [np.zeros((1, self.max_len - tlen, self.act_dim)), a[-1]],
            axis=1,
        )
        r[-1] = np.concatenate([np.zeros((1, self.max_len - tlen, 1)), r[-1]], axis=1)
        d[-1] = np.concatenate([np.ones((1, self.max_len - tlen)) * 2, d[-1]], axis=1)
        rtg[-1] = np.concatenate([np.zeros((1, self.max_len - tlen, 1)), rtg[-1]], axis=1) / self.scale
        #timesteps[-1] = np.concatenate([np.zeros((1, self.max_len - tlen)), timesteps[-1]], axis=1)
        mask.append(np.concatenate([np.zeros((1, self.max_len - tlen)), np.ones((1, tlen))], axis=1))

    def __call__(self, features):

        if (False):
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

        # a batch of dataset features
        s, a, r, d, rtg, timesteps, mask = [], [], [], [], [], [], []

        for feature in features:
            
            length = max(1,len(feature["rewards"]) - self.max_len)
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


class TrainableDT(DecisionTransformerModel):
    def __init__(self, config):
        super().__init__(config)
        self.agent_num_button_actions = AGENT_NUM_BUTTON_ACTIONS
        self.agent_num_camera_actions = AGENT_NUM_CAMERA_ACTIONS
        self.agent_esc_button = 2

        self.weight_button_actions = AGENT_NUM_CAMERA_ACTIONS/(AGENT_NUM_BUTTON_ACTIONS + AGENT_NUM_CAMERA_ACTIONS * 2)
        self.weight_camera_actions = AGENT_NUM_BUTTON_ACTIONS/(AGENT_NUM_BUTTON_ACTIONS + AGENT_NUM_CAMERA_ACTIONS * 2)
        self.weight_esc_button = 2/(AGENT_NUM_BUTTON_ACTIONS + AGENT_NUM_CAMERA_ACTIONS * 2)

        self.total_actions = self.agent_num_button_actions + self.agent_num_camera_actions + self.agent_esc_button

        self.normalized_weight_actions =  self.total_actions / self.agent_num_button_actions
        self.normalized_weight_camera = self.total_actions / self.agent_num_camera_actions
        self.normalized_weight_esc = self.total_actions / self.agent_esc_button

        self.action_size = self.agent_num_button_actions + self.agent_num_camera_actions + self.agent_esc_button
        self.predict_action = nn.Sequential(
            nn.Linear(config.hidden_size, self.action_size)
        )

    def forward(self, **kwargs):

        del kwargs["return_loss"]

        output = super().forward(**kwargs)
        
        return_dict = kwargs.get("return_dict", self.config.use_return_dict)

        action_targets = kwargs.get("actions")
        attention_mask = kwargs.get("attention_mask")

        flat_attention_mask = attention_mask.reshape(-1) > 0
  
        device = action_targets.device

        action_targets_button = action_targets[:, :, 0]
        action_targets_camera = action_targets[:, :, 1]
        action_targets_esc = action_targets[:, :, 2]

        if not return_dict:
            action_preds = output[1]
        else:
            action_preds = output.action_preds

        action_preds_button = action_preds[:, :, :self.agent_num_button_actions]
        action_preds_camera = action_preds[:, :, self.agent_num_button_actions:self.agent_num_button_actions + self.agent_num_camera_actions]   
        action_preds_esc = action_preds[:, :, self.agent_num_button_actions + self.agent_num_camera_actions:]
        action_preds_button = F.log_softmax(action_preds_button, dim=-1)
        action_preds_camera = F.log_softmax(action_preds_camera, dim=-1)
        action_preds_esc = F.log_softmax(action_preds_esc, dim=-1)

        action_preds_button = action_preds_button.reshape(-1, self.agent_num_button_actions)
        action_preds_camera = action_preds_camera.reshape(-1, self.agent_num_camera_actions)
        action_preds_esc = action_preds_esc.reshape(-1, self.agent_esc_button)

        action_targets_button = action_targets_button.reshape(-1).long()
        action_targets_camera = action_targets_camera.reshape(-1).long()
        action_targets_esc = action_targets_esc.reshape(-1).long()

        action_preds_button = action_preds_button[flat_attention_mask]
        action_preds_camera = action_preds_camera[flat_attention_mask]
        action_preds_esc = action_preds_esc[flat_attention_mask]

        action_targets_button = action_targets_button[flat_attention_mask]
        action_targets_camera = action_targets_camera[flat_attention_mask]
        action_targets_esc = action_targets_esc[flat_attention_mask]

        loss_button = F.nll_loss(action_preds_button, action_targets_button)
        loss_camera = F.nll_loss(action_preds_camera, action_targets_camera)
        loss_esc = F.nll_loss(action_preds_esc, action_targets_esc)

        loss = self.weight_button_actions * loss_button + self.weight_camera_actions * loss_camera + self.weight_esc_button * loss_esc
        #loss = loss_button + loss_camera + loss_esc

        return {"loss": loss}

    def original_forward(self, **kwargs):

        output = super().forward(**kwargs)
            
        return output


class SimpleBinaryClassifier(nn.Module):
    def __init__(self, input_dim):
        super(SimpleBinaryClassifier, self).__init__()
        self.fc1 = nn.Linear(1024, 1)  
    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))  
        return x