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
