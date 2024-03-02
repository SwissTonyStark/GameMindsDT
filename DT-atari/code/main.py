import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader,TensorDataset
import argparse

import math
import numpy as np
import matplotlib.pylab as plt

import gym
import d4rl_atari

from model import DecisionTransformer
from data import StateActionReturnDataset, create_dataset
from train import Trainer
from eval import evaluate_on_env

import warnings
warnings.filterwarnings('ignore')

def run(args):

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    hparams = {
        'game_name': args.game_name,
        'n_layer':6, # num of decoder blocks
        'n_head': 8, # atention heads
        'n_embd':128, # embedding vector size
        'context_length':args.context_length, # 30 in breakout, 50 in pong
        'dropout':0.1, # dropout value
        'act_voc':args.act_voc, # num of possible actions
        'batch_size': args.batch_size,
        'state_dim': 4*84*84, # 4 stacked (to capture movement) images (84,84)
        'max_timestep': 4096, # hardcoded default 4096, enough max steps in a trajectory
        'lr': 6e-4, # in original paper 1e-4
        'wt_decay':0.1, # in original paper 1e-4
        'warmup_steps':512*20,
        'target_reward': args.target_reward, 
        'max_epochs':5 
    }

    #### GAME ENVIRONMENT ########
    env = gym.make(hparams['game_name'], stack=True) # 4 stacked gray-scale images

    hparams['state_dim'] = env.observation_space.shape[0]*env.observation_space.shape[1]*env.observation_space.shape[2]
    print('action_space ', env.action_space)

    #### CREATE DATASET & DATALOADER #####
    obss, actions, done_idxs, rtgs, timesteps, maxTimestep = create_dataset(env)

    train_dataset = StateActionReturnDataset(obss, actions, done_idxs, rtgs, timesteps, hparams['context_length'])

    #### DECISION TRANSFORMER MODEL ########
    model = DecisionTransformer(
                    state_dim=hparams['state_dim'],
                    act_voc=hparams['act_voc'],
                    n_blocks=hparams['n_layer'],
                    h_dim=hparams['n_embd'],
                    context_len=hparams['context_length'],
                    n_heads=hparams['n_head'],
                    drop_p=hparams['dropout'],
                    max_timestep=hparams['max_timestep'], #default
                ).to(device)

    #### TRAINING ########
    optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=hparams['lr'],
            weight_decay=hparams['wt_decay'],
            betas=(0.9, 0.999)
            )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda steps: min((steps+1)/hparams['warmup_steps'], 1)
            )
    criterion = F.cross_entropy

    train_model = Trainer(model=model, optimizer=optimizer, criterion= criterion, scheduler=scheduler, device=device)
    train_model.train(train_dataset, hparams)

    #### EVALUATION ########
    evaluate_on_env(model, env, hparams, device, num_eval_ep=1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Model configs
    parser.add_argument('--game_name', type=str, default='qbert-expert-v4')
    parser.add_argument('--act_voc', type=int, default=6)
    parser.add_argument('--context_length', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--target_reward', type=int, default=2500)

    args = parser.parse_args()
    run(args)

