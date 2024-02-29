import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import numpy as np
from tqdm import tqdm

class Trainer():
    def __init__(self, 
        model, 
        optimizer, 
        criterion, 
        scheduler,
        device
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.device = device

    def train(self, dataloader, hparams):

            losses = []

            for epoch in range(hparams['max_epochs']):

                self.model.to(self.device)
                self.model.train()

                loader = DataLoader(dataloader, shuffle=True, pin_memory=True, 
                                    batch_size=hparams['batch_size'])

                pbar = tqdm(enumerate(loader), total=len(loader))
                for it, (states, actions, rtgs, timesteps) in pbar:

                    # place data on the correct device
                    states = states.to(self.device) # size([B, seq_len, state_dim]) state_dim = 28224
                    actions = actions.to(self.device) # size([B, seq_len, 1])
                    rtgs = rtgs.to(self.device) # size([B, seq_len, 1])
                    timesteps = timesteps.squeeze(-1).to(self.device) #   size([B, seq_len])
                    action_target = torch.clone(actions).detach().to(self.device)

                    # forward the model
                    with torch.set_grad_enabled(True):
                      action_preds= self.model.forward(
                                                    timesteps=timesteps,
                                                    states=states,
                                                    actions=actions,
                                                    returns_to_go=rtgs)

                      # compute loss
                      loss = self.criterion(action_preds.reshape(-1, action_preds.size(-1)), action_target.reshape(-1).long())
                      losses.append(loss.detach().cpu().item())

                      # backprop and update the parameters
                      self.optimizer.zero_grad()
                      loss.backward()
                      torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0) # 0.25
                      self.optimizer.step()
                      self.scheduler.step()

                    # report progress
                    pbar.set_description(f"epoch {epoch} iter {it}: train loss {loss.item():.5f}. lr {hparams['lr']:e}")

            return losses
