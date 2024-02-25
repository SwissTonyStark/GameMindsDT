import torch
import torch.nn as nn
import wandb

from datetime import datetime


class WandbLogger():

    def __init__(self, model: nn.Module):
        wandb.init(project="hands-on-monitoring")
        wandb.run.name = f'training-DT-{datetime.now().strftime("%Y%m%d-%H%M%S")}'
        wandb.watch(models=model)

    def log_training(
        self, 
        epoch: int,
        train_loss_avg: float, 
        val_loss_avg: float
    ):
        wandb.log({"Prediction/val_loss":val_loss_avg}, step=epoch)
        wandb.log({'Prediction/train_loss':train_loss_avg}, step=epoch)
