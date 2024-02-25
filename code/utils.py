import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader,TensorDataset

import math
import numpy as np
import seaborn as sns
import matplotlib.pylab as plt

#Especifico para el gym+dataset "D4RL_Pybullet"
import gym
import d4rl_pybullet

def get_episodes(terminals):
    terminals = terminals.astype('int32')
    #Las posiciones donde estan los Terminal=1
    if terminals[-1] == 0 : 
        terminals[-1] = 1  
    terminal_pos = np.where(terminals==1)[0]
    return terminal_pos.tolist(), len(terminal_pos)

def get_rtgs(t_positions, rewards):
    # Initialize the starting index of the sub-list in B
    start_idx = 0
    rtgs = []

    
    for t in t_positions:
        end_idx = t + 1
        sub_list = rewards[start_idx:end_idx]
        #print(sub_list)
        for i in range(0, len(sub_list)):
            rtgs.append(sum(sub_list[i+1:]))
        start_idx = end_idx
    return rtgs

def optimized_get_rtgs(t_positions, rewards):

    rewards = np.array(rewards, dtype=np.float64)
    t_positions = np.array(t_positions)

    cumsum_rewards = np.cumsum(rewards)
    
    # Initialize an array to hold the RTGs
    rtgs = np.array([], dtype=int)
    
    # Keep track of the start index of the sub-list in rewards
    start_idx = 0
    for end_idx in t_positions:
        
        segment_rtgs = cumsum_rewards[end_idx] - cumsum_rewards[start_idx:end_idx]
        segment_rtgs = np.append(segment_rtgs, 0)
        rtgs = np.concatenate((rtgs, segment_rtgs))
    
        start_idx = end_idx+1
    return rtgs.tolist()

def list_episodes(terminals_idxs):
    episode_ends = np.array(terminals_idxs)
    episode_starts=np.roll(episode_ends, shift=1) + 1
    episode_starts[0] = 0
    return list(zip(episode_starts, episode_ends +1))

def get_timesteps(episodes, size):
    # Initialize the array of timesteps
    arrayTimesteps = np.zeros(size, dtype=int)

    # List to hold the total steps per episode
    steps_per_episode = []

    # Generate timesteps for each episode
    for start, end in episodes:
        episode_length = end - start
        steps_per_episode.append(episode_length)
        arrayTimesteps[start:end] = np.arange(episode_length)
    return arrayTimesteps, steps_per_episode

def normalize_array(array):
    mean = np.mean(array, axis=0)
    std = np.std(array, axis=0)
    norm_array = (array - mean) / (std+1e-6)
    return norm_array, mean, std

def get_data_set(obs, actions, rewards, terminals, episodes):
    d_obs = [obs[start:end] for start, end in episodes]
    d_act = [actions[start:end] for start, end in episodes]
    d_rew = [rewards[start:end] for start, end in episodes]
    d_ter = [terminals[start:end] for start, end in episodes]

    r_obs = np.concatenate(d_obs, axis=0)
    r_act = np.concatenate(d_act, axis=0)
    r_rew = np.concatenate(d_rew, axis=0)
    r_ter = np.concatenate(d_ter, axis=0)

    return r_obs, r_act, r_rew, r_ter