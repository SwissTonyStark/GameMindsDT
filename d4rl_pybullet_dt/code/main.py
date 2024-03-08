import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gym
import logging
import d4rl_pybullet
import datetime

from torch.utils.data import DataLoader
from tabulate import tabulate

from model import DecisionTransformer
from data import DecisionTransformerDataset
from utils import *
from train import Trainer
from test_agent import TestAgent



"""
============================================
                  SEEDS                        
============================================
"""
#Numpy seed
np.random.seed(3344)

#Pytorch seed
torch.manual_seed(3344)



def main_loop(trigger_train, trigger_test, env_name, pretrained_file_name):

 
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
    
    """
    ============================================
           ENVIRONMENT CREATION & DATASET                        
    ============================================
    """

    # Environment creation
    print('Environment Selected: ',env_name)

    env = gym.make(env_name)
    
    # Obtain environment dataset
    dataset = env.get_dataset()

    raw_obs = dataset['observations'] # Observation data in a [N x dim_observation] numpy array 
    raw_actions = dataset['actions'] # Action data in [N x dim_action] numpy array
    raw_rewards = dataset['rewards'] # Reward data in a [N x 1] numpy array
    raw_terminals = dataset['terminals'] # Terminal flags in a [N x 1] numpy array

    """
    ============================================
                    EPISODES INFO                          
    ============================================
    """

    terminals_pos, num_episodes = get_episodes(raw_terminals)
    logging.info(f'Showing episodes information...')

    episodes = list_episodes(terminals_pos)
    timesteps, steps_per_episode = get_timesteps(episodes, len(raw_obs))
    max_timestep = np.max(steps_per_episode)
    min_timestep = np.min(steps_per_episode)
    mean_timestep= np.mean(steps_per_episode)

    ep_data =  [["Total episodes", num_episodes],
                ["Max duration", max_timestep],
                ["Min duration", min_timestep],
                ["Mean duration",mean_timestep]]
    col_head = ["", "Value"]

    tb = tabulate(ep_data, headers=col_head,tablefmt="grid")
    logging.info(f'\n{tb}')

    """
    ============================================
                PREPROCESS DATA                         
    ============================================
    """
    # Remove episodes wiht lees than mean_timestep

    rm_episode_idx = [idx for idx, mean in enumerate(steps_per_episode) if mean < mean_timestep]
    logging.info(f'Removing {len(rm_episode_idx)} eps out of {len(episodes)} eps...')
    logging.info(f'Remaining episoded should be {len(episodes) - len(rm_episode_idx)} eps.')
    final_episodes = [(start,end) for start, end in episodes if (end-start) >= mean_timestep]

    assert len(episodes) - len(rm_episode_idx) == len(final_episodes), "Error: Episodes size"

    observations, actions, rewards, terminals = get_data_set(raw_obs, raw_actions, raw_rewards, raw_terminals, final_episodes)
    logging.info(f'Final total samples: {observations.shape[0]} out of {raw_obs.shape[0]} original samples.')

    # Dataset normalizations

    observations,dataset_observations_mean,dataset_observations_std = normalize_array(observations)

    """
    ============================================    
            TRAINING CONFIGURATIONS_DICT  
                  ( RECOMMENDED )                    
    ============================================
    """     
    configurations_dict = {
        'hopper-bullet-medium-v0': {
            "h_dim": 128,  
            "num_heads": 1,
            "num_blocks": 3, 
            "context_len": 20,
            "batch_size": 64,
            "lr": 0.0001,
            "weight_decay": 0.0001,
            "mlp_ratio": 1,
            "dropout": 0.1,
            "train_epochs": 1500, 
            "rtg_target": 3600,
            "rtg_scale" :1,
            "constant_retrun_to_go" : True,
            "stochastic_start" : True,
            "num_eval_ep" : 10, 
            "max_eval_ep_len":250, 
            "num_test_ep":10,  
            "max_test_ep_len":1000,
            "state_mean" : dataset_observations_mean,
            "state_std" : dataset_observations_std, 
            "render" : False
        },
        

        'hopper-bullet-mixed-v0': {
            "h_dim": 128,  
            "num_heads": 1,
            "num_blocks": 3, 
            "context_len": 20,
            "batch_size": 64,
            "lr": 0.0001,
            "weight_decay": 0.0001,
            "mlp_ratio": 1,
            "dropout": 0.1,
            "train_epochs": 1500,
            "rtg_target": 3600,
            "rtg_scale" :1,
            "constant_retrun_to_go" : True,
            "stochastic_start" : True,
            "num_eval_ep" :10, 
            "max_eval_ep_len":250, 
            "num_test_ep":10,  
            "max_test_ep_len":1000,
            "state_mean" : dataset_observations_mean,
            "state_std" : dataset_observations_std, 
            "render" : False
        },

        'halfcheetah-bullet-medium-v0': {
            "h_dim": 256,  
            "num_heads": 4,
            "num_blocks": 3, 
            "context_len": 100,
            "batch_size": 128,
            "lr": 0.0001,
            "weight_decay": 0.0005,
            "mlp_ratio": 1,
            "dropout": 0.2,
            "train_epochs": 3000,
            "rtg_target": 6000,
            "rtg_scale" :1,
            "constant_retrun_to_go" : False,
            "stochastic_start" : True,
            "num_eval_ep" :3, 
            "max_eval_ep_len":250, 
            "num_test_ep":10,  
            "max_test_ep_len":1000,
            "state_mean" : dataset_observations_mean,
            "state_std" : dataset_observations_std, 
            "render" : False
        },

        'halfcheetah-bullet-mixed-v0': {
            "h_dim": 256,  
            "num_heads": 2,
            "num_blocks": 3, 
            "context_len": 20,
            "batch_size": 64,
            "lr": 0.0001,
            "weight_decay": 0.0001,
            "mlp_ratio": 1,
            "dropout": 0.1,
            "train_epochs": 3000,
            "rtg_target": 6000,
            "rtg_scale" :1,
            "constant_retrun_to_go" : True,
            "stochastic_start" : True,
            "num_eval_ep" :3, 
            "max_eval_ep_len":250, 
            "num_test_ep":10,  
            "max_test_ep_len":1000,
            "state_mean" : dataset_observations_mean,
            "state_std" : dataset_observations_std, 
            "render" : False
        },

        'ant-bullet-medium-v0': {
            "h_dim": 256,  
            "num_heads": 2,
            "num_blocks": 3, 
            "context_len": 100,
            "batch_size": 64,
            "lr": 0.0001,
            "weight_decay": 0.0001,
            "mlp_ratio": 1,
            "dropout": 0.1,
            "train_epochs": 3000,
            "rtg_target": 6000,
            "rtg_scale" :1,
            "constant_retrun_to_go" : False,
            "stochastic_start" : True,
            "num_eval_ep" :3, 
            "max_eval_ep_len":250, 
            "num_test_ep":10,  
            "max_test_ep_len":1000,
            "state_mean" : dataset_observations_mean,
            "state_std" : dataset_observations_std, 
            "render" : False
        },

        'ant-bullet-mixed-v0': {
            "h_dim": 256,  
            "num_heads": 2,
            "num_blocks": 3, 
            "context_len": 20,
            "batch_size": 64,
            "lr": 0.0001,
            "weight_decay": 0.0001,
            "mlp_ratio": 1,
            "dropout": 0.1,
            "train_epochs": 3000,
            "rtg_target": 6000,
            "rtg_scale" :1,
            "constant_retrun_to_go" : True,
            "stochastic_start" : True,
            "num_eval_ep" :3, 
            "max_eval_ep_len":250, 
            "num_test_ep":10,  
            "max_test_ep_len":1000,
            "state_mean" : dataset_observations_mean,
            "state_std" : dataset_observations_std, 
            "render" : False
        },

        'walker2d-bullet-medium-v0': { 
            "h_dim": 128,  
            "num_heads": 1,
            "num_blocks": 3, 
            "context_len": 20,
            "batch_size": 64,
            "lr": 0.0001,
            "weight_decay": 0.0001,
            "mlp_ratio": 1,
            "dropout": 0.1,
            "train_epochs": 1500,
            "rtg_target": 5000,
            "rtg_scale" :1,
            "constant_retrun_to_go" : True,
            "stochastic_start" : True,
            "num_eval_ep" :10, 
            "max_eval_ep_len":250, 
            "num_test_ep":10,  
            "max_test_ep_len":1000,
            "state_mean" : dataset_observations_mean,
            "state_std" : dataset_observations_std, 
            "render" : False
        },

        'walker2d-bullet-mixed-v0': { 
            "h_dim": 128,  
            "num_heads": 1,
            "num_blocks": 3, 
            "context_len": 20,
            "batch_size": 64,
            "lr": 0.0001,
            "weight_decay": 0.0001,
            "mlp_ratio": 1,
            "dropout": 0.1,
            "train_epochs": 1500,
            "rtg_target": 5000,
            "rtg_scale" :1,
            "constant_retrun_to_go" : True,
            "stochastic_start" : True,
            "num_eval_ep" :10, 
            "max_eval_ep_len":250, 
            "num_test_ep":10,  
            "max_test_ep_len":1000,
            "state_mean" : dataset_observations_mean,
            "state_std" : dataset_observations_std, 
            "render" : False
        }
    } 


    """
    ============================================
                    DATA SPLIT                         
    ============================================
    """

    f_ter_idx, n_eps = get_episodes(terminals)
    f_episodes = list_episodes(f_ter_idx)

    np.random.shuffle(f_episodes)

    train_size = int(n_eps * 0.8) # remaining 0.2 for validation

    train_episodes = f_episodes[:train_size]
    val_episodes = f_episodes[train_size:]

    train_obs, train_act, train_rew, train_ter = get_data_set(observations, actions, rewards, terminals, train_episodes)
    val_obs, val_act, val_rew, val_ter = get_data_set(observations, actions, rewards, terminals, val_episodes)

    train_terminals_idx, _ = get_episodes(train_ter)
    train_rtgs = optimized_get_rtgs(train_terminals_idx, train_rew)
    t_episodes = list_episodes(train_terminals_idx)
    train_timesteps, _ = get_timesteps(t_episodes, len(train_obs))

    val_terminals_idx, _ = get_episodes(val_ter)
    val_rtgs = optimized_get_rtgs(val_terminals_idx, val_rew)
    v_episodes = list_episodes(val_terminals_idx)
    val_timesteps, _ = get_timesteps(v_episodes, len(val_obs))


    hparams = configurations_dict[env_name]

    train_dataset = DecisionTransformerDataset(train_obs, train_act, train_timesteps, train_rtgs, train_terminals_idx, hparams['context_len'])
    train_loader = DataLoader(train_dataset, batch_size=hparams['batch_size'], shuffle=True)

    val_dataset = DecisionTransformerDataset(val_obs, val_act, val_timesteps, val_rtgs, val_terminals_idx, hparams['context_len'])
    val_loader = DataLoader(val_dataset, batch_size=hparams['batch_size'], shuffle=True)

    model_cfg = {
        "state_dim": env.observation_space.shape[0],
        "act_dim": env.action_space.shape[0], # act_dim=Contextlength?
        "h_dim": hparams['h_dim'],  #embed_dim
        "num_heads": hparams['num_heads'],
        "num_blocks": hparams['num_blocks'],
        "context_len": hparams['context_len'],
        "max_timesteps": max_timestep,
        "mlp_ratio": hparams['mlp_ratio'],
        "dropout": hparams['dropout']
    }

    """
    ============================================
            TRAIN DECISION TRANSFORMER                         
    ============================================
    """
    if trigger_train:
        
        model_dt = DecisionTransformer(**model_cfg).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model_dt.parameters(),weight_decay=hparams['weight_decay'], lr=hparams['lr'])


        train_model = Trainer(model=model_dt,
                            env=env,
                            env_name=env_name, 
                            optimizer=optimizer, 
                            criterion= criterion, 
                            device=device, 
                            hyperparameters=hparams)
        train_model.train(train_loader=train_loader, val_loader=val_loader)

        # Now save the artifacts of the training
        file_name = f'weights-{env.unwrapped.spec.id}.pt'
        save_dir = os.getcwd()+ f'\checkpoints\{file_name}' 

        # Create the directory if don't exist   
        os.makedirs("checkpoints", exist_ok=True)  

        # Check existing files to avoid overwriting 
        count = 1
        while os.path.exists(save_dir):
            save_dir = os.path.join("checkpoints", f'state-{env.unwrapped.spec.id}_{count}.pt')
            count += 1

        logging.info(f"Saving checkpoint to {save_dir}...")

        # Save the parameters,weights and biases, optimizers, environment name and model's config.
        checkpoint = {
            "model_state_dict": model_dt.state_dict(),  
            "optimizer_state_dict": optimizer.state_dict(),
            "env_name": env.unwrapped.spec.id,
            "config": model_cfg
        }
        torch.save(checkpoint, save_dir)

        print("Training Completed Succesfully!")


    """
    ============================================
            TEST DECISION TRANSFORMER                         
    ============================================
    """

    if trigger_test:

        
        load_dir = os.path.join(os.getcwd(),'checkpoints',pretrained_file_name)
        print("Weights will be loaded from: ", load_dir)

        # Load the weights and config from the pretrained Decision Transformer model (.pt file)
        checkpoint = torch.load(load_dir, map_location=torch.device('cuda'))
        print("Weights loaded succesfully")
        #Instantiate the Decision Transformer Model using the .pt file
        agent = DecisionTransformer(**checkpoint['config']).to(device)
        agent.load_state_dict(checkpoint['model_state_dict'])
        print("Model Initialization Succesful")

        test_agent = TestAgent(agent,device,env,env_name)

        test_results = test_agent.test(context_len = hparams["context_len"],
                        rtg_target = hparams["rtg_target"], 
                        rtg_scale = hparams["rtg_scale"],
                        constant_retrun_to_go = hparams["constant_retrun_to_go"],
                        stochastic_start = hparams["stochastic_start"],
                        num_test_ep = hparams["num_test_ep"],
                        max_test_ep_len = hparams["max_test_ep_len"],
                        state_mean = hparams["state_mean"], 
                        state_std = hparams["state_std"], 
                        render = hparams["render"]
                        )

        print ("Test results:", test_results)

        print("Test Completed Succesfully!")
          
   


if __name__ == "__main__":
    clear_console()
    print_logo()
    trigger_train, trigger_test, env_name, pretrained_file_name = navigate_main_menu() 
    
    if trigger_train or trigger_test:
        main_loop(trigger_train, trigger_test, env_name, pretrained_file_name)

     


print ("\n","/"*60)
print (" "*20,"Execution completed at:",datetime.datetime.now().strftime("%H:%MH%S"))
print ("/"*60)


