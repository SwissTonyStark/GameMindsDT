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

import logging

from model import DecisionTransformer
from data import MyDataset

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
logging.basicConfig(level=logging.INFO)

def get_episodes():
    terminals = dataset['terminals'].astype('int32')
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

def get_timestep(terminal_pos):
    start_index = 0
    arrayTimesteps = np.zeros(len(dataset['rewards']), dtype=int)
    for i in terminal_pos:
        arrayTimesteps[start_index:(i+1)] = np.arange((i+1) - start_index)
        start_index = i
    return arrayTimesteps

def list_episodes(self, terminals):
    episode_ends = np.array(terminals)
    episode_starts=np.roll(episode_ends, shift=1) + 1
    episode_starts[0] = 0
    return list(zip(episode_starts, episode_ends +1))



hparams = {
    "h_dim": 128,  #embed_dim
    "num_heads": 2,
    "num_blocks": 4,
    "context_len": 30,
    "lr": 0.001,
    "mlp_ratio": 4,
    "dropout": 0.1,
    "epochs": 50
}
env = gym.make('hopper-bullet-mixed-v0')
dataset = env.get_dataset()

arrayObservations = dataset['observations'] # Observation data in a [N x dim_observation] numpy array  ==> Para 'hopper-bullet-mixed-v0" = [59345 x 15]
arrayActions = dataset['actions'] # Action data in [N x dim_action] numpy array ==> Para 'hopper-bullet-mixed-v0" = [59345 x 3]
arrayRewards = dataset['rewards'] # Reward data in a [N x 1] numpy array ==> Para 'hopper-bullet-mixed-v0" = [59345 x 1]
arrayTerminals = dataset['terminals'] # Terminal flags in a [N x 1] numpy array ==> Para 'hopper-bullet-mixed-v0" = [59345 x 1]

terminals_pos, num_episodes = get_episodes()
logging.info(f'TOTAL EPISODES: {num_episodes}')

rtgs = optimized_get_rtgs(terminals_pos, dataset['rewards'])
timesteps = get_timestep(terminals_pos)
max_timesteps = max(timesteps) + 1
logging.info(f'MAX TIMESTEP OF A EPISODE: {max_timesteps}')


blocks = hparams['context_len']
dataset = MyDataset(arrayObservations, arrayActions, timesteps, rtgs, terminals_pos, blocks)
DTDataLoader = DataLoader(dataset, batch_size=1, shuffle=False)

state_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]

model_cfg = {
    "state_dim": state_dim,
    "act_dim": act_dim, # act_dim=Contextlength?
    "h_dim": hparams['h_dim'],  #embed_dim
    "num_heads": hparams['num_heads'],
    "num_blocks": hparams['num_blocks'],
    "context_len": hparams['context_len'],
    "max_timesteps": max_timesteps,
    "mlp_ratio": hparams['mlp_ratio'],
    "dropout": hparams['dropout']
}

model_dt = DecisionTransformer(**model_cfg).to(device)


criterion = nn.MSELoss()
optimizer = optim.Adam(model_dt.parameters(), lr=hparams['lr'])

num_epochs = hparams['epochs']
timestep = 0
max_timesteps = max_timesteps #Calculated according to the longest episode in the dataset/env loaded.
for epoch in range(num_epochs):
    total_loss = 0.0  # Inicializar la pérdida total para el epoch
    model_dt.train()    # set model to trining mode
    # Iteración sobre los lotes de datos

    for states, actions, rtgs, steps, padd_mask in DTDataLoader:

        states = states.to(device)
        actions = actions.to(device)
        steps = steps.to(device)
        rtgs = rtgs.to(device)
        padd_mask = padd_mask.to(device)
        action_target = torch.clone(actions).detach().to(device)
        # Paso 1: Reiniciar los gradientes
        #timestep += 1 ==> No es necesario para el training, solo para evaluation
        optimizer.zero_grad()

        # Paso 2: Propagación hacia adelante (Forward pass)
        _, _, act_preds = model_dt(steps, states, actions, rtgs) # timestep, max_timesteps, states, actions, returns_to_go
        #outputs = model(batch_obs)

        # Paso 3: Calcular la pérdida
        act_preds = act_preds[padd_mask]
        actions = actions[padd_mask]
        loss = criterion(act_preds, actions)

        # Paso 4: Propagación hacia atrás (Backward pass)
        loss.backward()

        # Paso 5: Actualización de los parámetros del modelo
        optimizer.step()

        # Sumar la pérdida del batch a la pérdida total del epoch
        total_loss += loss.item()

    # Calcular la pérdida media del epoch
    epoch_loss = total_loss / len(DTDataLoader)

    # Imprimir la pérdida media del epoch
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')

    # Paso 6 (Opcional): Evaluación del modelo en un conjunto de datos de evaluación
    # Aquí puedes agregar código para evaluar el modelo en un conjunto de datos de evaluación si lo tienes disponible

# Paso 7 (Opcional): Visualización de resultados o métricas de rendimiento
# Aquí puedes agregar código para mostrar otras métricas de rendimiento que desees analizar