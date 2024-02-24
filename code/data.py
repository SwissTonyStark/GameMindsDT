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

class DecisionTransformerDataset(Dataset):
    def __init__(self, observations, actions, steps, rtgs, terminals, blocks):
        self.observations = observations
        self.actions = actions
        self.steps = steps
        self.rtgs = rtgs
        self.terminals = terminals
        self.blocks = blocks
        self.episodes = self._determine_episodes(terminals)

    def _determine_episodes(self, terminals):
        episode_ends = np.array(terminals)
        episode_starts=np.roll(episode_ends, shift=1) + 1
        episode_starts[0] = 0
        return list(zip(episode_starts, episode_ends +1))

    def __len__(self):
        return len(self.terminals)

    def __getitem__(self, idx):
        # to avoid blocks in between of 2 trajectories, if the idx is too close to the end of a trajectory, re-position
        # the idx to a block_size away to the end of the trajectory []
        start, end = self.episodes[idx]

        episode_length = end - start 

        # Sample a start point for the sequence within the episode
        if self.blocks <= episode_length:
            seq_start = np.random.randint(start, end - self.blocks + 1)
        else:
            seq_start = start

        states = np.zeros((self.blocks, self.observations.shape[1]))
        actions = np.zeros((self.blocks, self.actions.shape[1]))
        rtgs = np.zeros((self.blocks,))
        steps = np.zeros((self.blocks,))
        padding_mask = np.zeros((self.blocks,))

        # if we actually can get a subsequence of the episode then the end of the sequence is
        # start + sequence lenght, otherwise, the end of the sequence is the end of the episode
        # this is done to avoid using conditional ifs.
        act_seq_len = min(self.blocks, end - seq_start)
        seq_end = seq_start + act_seq_len

        states[:act_seq_len]    = (self.observations[seq_start:seq_end])
        actions[:act_seq_len]   = (self.actions[seq_start:seq_end])
        rtgs[:act_seq_len]      = (self.rtgs[seq_start:seq_end])
        steps[:act_seq_len]     = (self.steps[seq_start:seq_end])
        padding_mask[:act_seq_len]=1

        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rtgs = torch.FloatTensor(rtgs).unsqueeze(dim=-1) #T x 1 
        steps = torch.tensor(steps, dtype=torch.int)
        padding_mask = torch.tensor(padding_mask, dtype=torch.bool)

        return states, actions, rtgs, steps, padding_mask