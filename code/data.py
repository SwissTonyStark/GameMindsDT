import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader,TensorDataset

import math
import numpy as np
#import seaborn as sns
import matplotlib.pylab as plt

#Especifico para el gym+dataset "D4RL_Pybullet"
import gym
import d4rl_pybullet

from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, observations, actions, steps, rtgs, terminals, blocks):
        self.observations = observations
        self.actions = actions
        self.steps = steps
        self.rtgs = rtgs
        self.terminals = terminals
        self.blocks = blocks

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, idx):
        # to avoid blocks in between of 2 trajectories, if the idx is too close to the end of a trajectory, re-position
        # the idx to a block_size away to the end of the trajectory
        episode_ends = np.array(self.terminals)
        episode_starts=np.roll(episode_ends, shift=1) + 1
        episode_starts[0] = 0

        print("episode start", len(episode_starts))
        print("episode end", len(episode_ends))
        print(idx)
        start, end = list(zip(episode_starts, episode_ends +1))[idx]

        episode_length = end - start

        # Sample a start point for the sequence within the episode
        if episode_length >= self.blocks:
            seq_start = np.random.randint(start, end - self.blocks + 1)
            seq_end = seq_start + self.blocks
            n_padding = 0
        else:
            seq_start = start
            seq_end = start + episode_length - 1
            n_padding = self.blocks - episode_length + 1
        

        states = (self.observations[seq_start : seq_end])
        actions = (self.actions[seq_start : seq_end])
        rtgs = (self.rtgs[seq_start : seq_end])
        steps = (self.steps[seq_start : seq_end])
        
        if n_padding > 0:
            padding = np.zeros(n_padding)

            states = np.concatenate(states, padding)
            actions = np.concatenate(actions, padding)
            rtgs = np.concatenate(rtgs, padding)
            steps = np.concatenate(steps, padding)
        
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rtgs = torch.FloatTensor(rtgs).unsqueeze(dim=-1) #T x 1 
        steps = torch.tensor(steps, dtype=torch.int)

        return states, actions, rtgs, steps