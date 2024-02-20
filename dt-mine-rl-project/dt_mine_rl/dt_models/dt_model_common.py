import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import numpy as np
import os
import csv

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
    def __init__(self, button_act_csv_path, number_of_acts=256):
        self.number_of_acts = number_of_acts
        self.act_to_index = {}
        self.index_to_act = {}

        self.load_and_prepare(button_act_csv_path)
    
    def load_and_prepare(self, csv_file_path):
        with open(csv_file_path, mode='r') as csvfile:
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
    
