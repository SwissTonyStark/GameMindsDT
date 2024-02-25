import torch
import torch.nn as nn
from torch.utils.data import DataLoader,TensorDataset

import numpy as np

import logging

from wandb_logger import WandbLogger

class Trainer():
    def __init__(self, 
        model, 
        optimizer, 
        criterion, 
        device, 
        wandb_log
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.wandb = wandb_log
    
    def __train_step(self, train_loader: DataLoader):
        train_loss = 0
        for states, actions, rtgs, steps, padd_mask in train_loader:

            states, actions, steps, rtgs, padd_mask = (x.to(self.device) for x in [states, actions, steps, rtgs, padd_mask])
            action_target = torch.clone(actions).detach().to(self.device)

            self.optimizer.zero_grad()

            _, _, act_preds, selfattn_ws = self.model(steps, states, actions, rtgs) # timestep, max_timesteps, states, actions, returns_to_go

            act_preds = act_preds[padd_mask]
            actions = actions[padd_mask]
            loss = self.criterion(act_preds, actions)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
        return train_loss

    def __val_step(self, val_loader: DataLoader):
        val_loss = 0.0
        with torch.no_grad():  # No gradgients tracking
            for states, actions, rtgs, steps, padd_mask in val_loader:
                states, actions, steps, rtgs, padd_mask = (x.to(self.device) for x in [states, actions, steps, rtgs, padd_mask])
                action_target = torch.clone(actions).detach().to(self.device)

                _, _, act_preds, _ = self.model(steps, states, actions, rtgs)

                loss = self.criterion(act_preds, action_target)  
                val_loss += loss.item()
        return val_loss

    def train(self, num_epochs, train_loader: DataLoader, val_loader: DataLoader):
        for epoch in range(num_epochs):
            self.model.train()
            train_loss = self.__train_step(train_loader)
            self.model.eval()
            val_loss = self.__val_step(val_loader)

            epoc_val_loss = val_loss / len(val_loader.dataset) 
            epoch_train_loss = train_loss / len(val_loader.dataset)

            self.wandb.log_training(epoch=epoch, train_loss_avg=epoch_train_loss, val_loss_avg=epoc_val_loss)
            # Imprimir la pérdida media del epoch
            print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {epoch_train_loss:.4f}, Validation Loss: {epoc_val_loss:.4f}')
