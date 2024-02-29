import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

import math
import numpy as np
import matplotlib.pylab as plt





def create_dataset(env):

          # GET THE DATA !!
          dataset = env.get_dataset()

          obs_data = dataset['observations'] # observation data in (1000000, 1, 84, 84)
          action_data = dataset['actions'] # action data in (1000000,)
          reward_data = dataset['rewards'] # reward data in (1000000,)
          terminal_data = dataset['terminals'] # terminal flags in (1000000,)

          obs_data = dataset['observations'] # observation data in (1000000, 1, 84, 84)
          action_data = dataset['actions'] # action data in (1000000,)
          reward_data = dataset['rewards'] # reward data in (1000000,)
          terminal_data = dataset['terminals'] # terminal flags in (1000000,)


          plt.imshow(obs_data[1000][0])
          plt.show()

          terminal_pos = np.where(terminal_data==1)[0]
          terminal_data = None # de-allocate mem
          print("num episodes ", terminal_pos.shape)

          # -- create reward-to-go dataset
          start_index = 0
          rtg = np.zeros_like(reward_data)
          for i in terminal_pos:
              curr_traj_returns = reward_data[start_index:i]
              reward_acum = 0
              for j in range(i-1, start_index-1, -1): # start from i-1
                  reward_acum += reward_data[j]
                  rtg[j] = reward_acum
              start_index = i
          print('max rtg is %d' % max(rtg))

          reward_data = None

          # -- create timestep dataset ******************************
          start_index = 0
          timesteps = np.zeros(len(action_data), dtype=int)
          for i in terminal_pos:
              timesteps[start_index:i] = np.arange(i - start_index)
              start_index = i

          max_timestep = max(timesteps)
          print('max timestep is %d' % max_timestep)
          print("***** data loaded **********")


          return obs_data, action_data, terminal_pos, rtg, timesteps, max_timestep


# Class that picks up a block of data from the dataset
class StateActionReturnDataset(Dataset):

        def __init__(self, data, actions, done_idxs, rtgs, timesteps, context_length):
            self.context_length = context_length
            self.data = data
            self.actions = actions
            self.done_idxs = done_idxs
            self.rtgs = rtgs
            self.timesteps = timesteps

        def __len__(self):
            return len(self.data) - self.context_length * 3

        def __getitem__(self, idx):
            # to avoid blocks in between of 2 trajectories, if the idx is too close to the end of a trajectory, re-position
            # the idx to a context_length away to the end of the trajectory
            done_idx = idx + self.context_length
            for i in self.done_idxs:
                if i > idx: # first done_idx greater than idx
                    done_idx = min(int(i), done_idx)
                    break
            idx = done_idx - self.context_length
            states = torch.tensor(np.array(self.data[idx:done_idx]), dtype=torch.float32).reshape(self.context_length, -1) # (self.context_length, 4*84*84)
            states = states / 255. # normalize data
            actions = torch.tensor(self.actions[idx:done_idx], dtype=torch.long).unsqueeze(1) # (self.context_length, 1)
            rtgs = torch.tensor(self.rtgs[idx:done_idx], dtype=torch.float32).unsqueeze(1)
            timesteps = torch.tensor(self.timesteps[idx:done_idx], dtype=torch.int64).unsqueeze(1) # (1,1)

            return states, actions, rtgs, timesteps