import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader,TensorDataset

import math
import numpy as np
#import seaborn as sns
import matplotlib.pylab as plt
import pandas as pd
from tabulate import tabulate

#Especifico para el gym+dataset "D4RL_Pybullet"
import gym
import d4rl_pybullet

import logging

from model import DecisionTransformer
from data import DecisionTransformerDataset
from utils import *
from wandb_logger import WandbLogger
from train import Trainer

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

env = gym.make('hopper-bullet-mixed-v0')
dataset = env.get_dataset()

raw_obs = dataset['observations'] # Observation data in a [N x dim_observation] numpy array  ==> Para 'hopper-bullet-mixed-v0" = [59345 x 15]
raw_actions = dataset['actions'] # Action data in [N x dim_action] numpy array ==> Para 'hopper-bullet-mixed-v0" = [59345 x 3]
raw_rewards = dataset['rewards'] # Reward data in a [N x 1] numpy array ==> Para 'hopper-bullet-mixed-v0" = [59345 x 1]
raw_terminals = dataset['terminals'] # Terminal flags in a [N x 1] numpy array ==> Para 'hopper-bullet-mixed-v0" = [59345 x 1]

"""
============================================
                EPISODES INFO                          
============================================
"""

terminals_pos, num_episodes = get_episodes(raw_terminals)
logging.info(f'Showing episodes information...')

episodes = list_episodes(terminals_pos)
timesteps, steps_per_episode = get_timesteps(episodes, len(raw_obs))
max_timestep = np.max(steps_per_episode)
min_timestep = np.min(steps_per_episode)
mean_timestep= np.mean(steps_per_episode)

ep_data =  [["Total episodes", num_episodes],
            ["Max duration", max_timestep],
            ["Min duration", min_timestep],
            ["Mean duration",mean_timestep]]
col_head = ["", "Value"]

tb = tabulate(ep_data, headers=col_head,tablefmt="grid")
logging.info(f'\n{tb}')

"""
============================================
               PREPROCESS DATA                         
============================================
"""
### Remove episodes wiht lees than mean_timestep

rm_episode_idx = [idx for idx, mean in enumerate(steps_per_episode) if mean < mean_timestep]
logging.info(f'Removing {len(rm_episode_idx)} eps out of {len(episodes)} eps...')
logging.info(f'Remaining episoded should be {len(episodes) - len(rm_episode_idx)} eps.')
final_episodes = [(start,end) for start, end in episodes if (end-start) >= mean_timestep]

assert len(episodes) - len(rm_episode_idx) == len(final_episodes), "Error: Episodes size"

observations, actions, rewards, terminals = get_data_set(raw_obs, raw_actions, raw_rewards, raw_terminals, final_episodes)
logging.info(f'Final total samples: {observations.shape[0]} out of {raw_obs.shape[0]} original samples.')

### Normalization

observations,_,_ = normalize_array(observations)

"""
============================================
                 DATA SPLIT                         
============================================
"""

f_ter_idx, n_eps = get_episodes(terminals)
f_episodes = list_episodes(f_ter_idx)

np.random.shuffle(f_episodes)

train_size = int(n_eps * 0.8) # remaining 0.2 for validation

train_episodes = f_episodes[:train_size]
val_episodes = f_episodes[train_size:]

train_obs, train_act, train_rew, train_ter = get_data_set(observations, actions, rewards, terminals, train_episodes)
val_obs, val_act, val_rew, val_ter = get_data_set(observations, actions, rewards, terminals, val_episodes)

train_terminals_idx, _ = get_episodes(train_ter)
train_rtgs = optimized_get_rtgs(train_terminals_idx, train_rew)
t_episodes = list_episodes(train_terminals_idx)
train_timesteps, _ = get_timesteps(t_episodes, len(train_obs))

val_terminals_idx, _ = get_episodes(val_ter)
val_rtgs = optimized_get_rtgs(val_terminals_idx, val_rew)
v_episodes = list_episodes(val_terminals_idx)
val_timesteps, _ = get_timesteps(v_episodes, len(val_obs))

hparams = {
    "h_dim": 128,  #embed_dim
    "num_heads": 2,
    "num_blocks": 4,
    "context_len": 30,
    "batch_size": 32,
    "lr": 0.001,
    "mlp_ratio": 4,
    "dropout": 0.1,
    "epochs": 50
}

context_len = hparams['context_len']
train_dataset = DecisionTransformerDataset(train_obs, train_act, train_timesteps, train_rtgs, train_terminals_idx, context_len)
train_loader = DataLoader(train_dataset, batch_size=hparams['batch_size'], shuffle=False)

val_dataset = DecisionTransformerDataset(val_obs, val_act, val_timesteps, val_rtgs, val_terminals_idx, context_len)
val_loader = DataLoader(val_dataset, batch_size=hparams['batch_size'], shuffle=False)

model_cfg = {
    "state_dim": env.observation_space.shape[0],
    "act_dim": env.action_space.shape[0], # act_dim=Contextlength?
    "h_dim": hparams['h_dim'],  #embed_dim
    "num_heads": hparams['num_heads'],
    "num_blocks": hparams['num_blocks'],
    "context_len": hparams['context_len'],
    "max_timesteps": max_timestep,
    "mlp_ratio": hparams['mlp_ratio'],
    "dropout": hparams['dropout']
}

model_dt = DecisionTransformer(**model_cfg).to(device)

wandb_log = WandbLogger(model=model_dt)
criterion = nn.MSELoss()
optimizer = optim.Adam(model_dt.parameters(), lr=hparams['lr'])

train_model = Trainer(model=model_dt, optimizer=optimizer, criterion= criterion, device=device, wandb_log=wandb_log)
train_model.train(hparams['epochs'], train_loader=train_loader, val_loader=val_loader)

# Now save the artifacts of the training
savedir = f'./checkpoints/state-{env.unwrapped.spec.id}.pt'
logging.info(f"Saving checkpoint to {savedir}...")
# We can save everything we will need later in the checkpoint.
checkpoint = {
    "model_state_dict": model_dt.cpu().state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "env_name": env.unwrapped.spec.id
}
torch.save(checkpoint, savedir)