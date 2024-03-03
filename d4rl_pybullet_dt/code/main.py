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
               USER INTERFACE                         
============================================
"""

""" if __name__ == "__main__":
    clear_console()
    print_logo()
    navegar_menu_principal() """




def main_loop():

    env_id = 0

    while True:
       
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
        """
        ============================================
                        ENVIRONMENT                         
        ============================================
        """
        # Definition Box space bounds with float32 precision
        low = np.array([-1.0, -2.0, -3.0], dtype=np.float32)
        high = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        #Create Box space with the specified bounds
        box_space = gym.spaces.Box(low=low, high=high)

        env_names = [

            # Hopper Agent
            #'hopper-bullet-random-v0', #==> tiene valors seguramente NaN
            'hopper-bullet-medium-v0',
            'hopper-bullet-mixed-v0',

            # HalfCheetah Agent
            #'halfcheetah-bullet-random-v0',#==> tiene valors seguramente NaN
            'halfcheetah-bullet-medium-v0',
            'halfcheetah-bullet-mixed-v0',

             # Ant Agent
            #'ant-bullet-random-v0',#==> tiene valors seguramente NaN
            'ant-bullet-medium-v0',
            'ant-bullet-mixed-v0',

            # Walker2D Agent
            #'walker2d-bullet-random-v0',#==> tiene valors seguramenteNaN
            'walker2d-bullet-medium-v0',
            'walker2d-bullet-mixed-v0'
        ]

        env_name = env_names[env_id]
        print(env_name)

        # Hopper Agent
        #env_name = 'hopper-bullet-random-v0'
        #env_name = 'hopper-bullet-medium-v0'
        #env_name = 'hopper-bullet-mixed-v0'

        # HalfCheetah Agent
        #env_name = 'halfcheetah-bullet-random-v0'
        #env_name = 'halfcheetah-bullet-medium-v0'
        #env_name = 'halfcheetah-bullet-mixed-v0'

        # Ant Agent
        #env_name = 'ant-bullet-random-v0'
        #env_name = 'ant-bullet-medium-v0'
        #env_name = 'ant-bullet-mixed-v0'

        # Walker2D Agent
        #env_name = 'walker2d-bullet-random-v0'
        #env_name = 'walker2d-bullet-medium-v0'
        #env_name = 'walker2d-bullet-mixed-v0'


        env = gym.make(env_name)

        dataset = env.get_dataset()

        raw_obs = dataset['observations'] # Observation data in a [N x dim_observation] numpy array  ==> Para 'hopper-bullet-mixed-v0" = [59345 x 15]
        raw_actions = dataset['actions'] # Action data in [N x dim_action] numpy array ==> Para 'hopper-bullet-mixed-v0" = [59345 x 3]
        raw_rewards = dataset['rewards'] # Reward data in a [N x 1] numpy array ==> Para 'hopper-bullet-mixed-v0" = [59345 x 1]
        raw_terminals = dataset['terminals'] # Terminal flags in a [N x 1] numpy array ==> Para 'hopper-bullet-mixed-v0" = [59345 x 1]

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
        ### Remove episodes wiht lees than mean_timestep

        rm_episode_idx = [idx for idx, mean in enumerate(steps_per_episode) if mean < mean_timestep]
        logging.info(f'Removing {len(rm_episode_idx)} eps out of {len(episodes)} eps...')
        logging.info(f'Remaining episoded should be {len(episodes) - len(rm_episode_idx)} eps.')
        final_episodes = [(start,end) for start, end in episodes if (end-start) >= mean_timestep]

        assert len(episodes) - len(rm_episode_idx) == len(final_episodes), "Error: Episodes size"

        observations, actions, rewards, terminals = get_data_set(raw_obs, raw_actions, raw_rewards, raw_terminals, final_episodes)
        logging.info(f'Final total samples: {observations.shape[0]} out of {raw_obs.shape[0]} original samples.')

        ### Normalization

        observations,dataset_observations_mean,dataset_observations_std = normalize_array(observations)

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

        hparams = {
            "h_dim": 128,  #embed_dim
            "num_heads": 8,
            "num_blocks": 4, 
            "context_len": 15,
            "batch_size": 128,
            "lr": 0.001,
            "mlp_ratio": 4,
            "dropout": 0.1,
            "train_epochs": 2500, #2500
            "rtg_target": 1500,
            "rtg_scale" :1,
            "constant_retrun_to_go" : True,
            "num_eval_ep" :10, #10
            "max_eval_ep_len":150, #150
            "num_test_ep":10,  #10
            "max_test_ep_len":1000, #1000
            "state_mean" : dataset_observations_mean,
            "state_std" : dataset_observations_std, 
            "render" : False
        }


        train_dataset = DecisionTransformerDataset(train_obs, train_act, train_timesteps, train_rtgs, train_terminals_idx, hparams['context_len'])
        train_loader = DataLoader(train_dataset, batch_size=hparams['batch_size'], shuffle=False)

        val_dataset = DecisionTransformerDataset(val_obs, val_act, val_timesteps, val_rtgs, val_terminals_idx, hparams['context_len'])
        val_loader = DataLoader(val_dataset, batch_size=hparams['batch_size'], shuffle=False)

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
        TRAIN = True

        if TRAIN:
            #wandb_log=wandb.init(project="Train DecisionTransformer")

            # Capture a dictionary of hyperparameters with config
            #wandb.config = {'context_length': hparams['context_len'],'learning_rate': hparams['lr'],'training_epochs': hparams['epochs']}

            model_dt = DecisionTransformer(**model_cfg).to(device)
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model_dt.parameters(), lr=hparams['lr'])


            train_model = Trainer(model=model_dt,
                                env=env,
                                env_name=env_name, 
                                optimizer=optimizer, 
                                criterion= criterion, 
                                device=device, 
                                hyperparameters=hparams)
            train_model.train(train_loader=train_loader, val_loader=val_loader)

            # Now save the artifacts of the training
            # O.A 26/02/2024 -> Hago cambios para que el codigo no pete si no existe el directorio.
            file_name = f'weights-{env.unwrapped.spec.id}.pt'
            savedir = os.getcwd()+ f'\checkpoints\{file_name}' 

            # Create the directory if don't exist   #O.A 26.02.2024
            os.makedirs("checkpoints", exist_ok=True)  

            # Check existing files to avoid ovewritting   #O.A 26.02.2024
            count = 1
            while os.path.exists(savedir):
                savedir = os.path.join("checkpoints", f'state-{env.unwrapped.spec.id}_{count}.pt')
                count += 1

            logging.info(f"Saving checkpoint to {savedir}...")

            # Save the parameters,weights and biases, optimizers, environment name and model's config.
            checkpoint = {
                "model_state_dict": model_dt.state_dict(),  
                "optimizer_state_dict": optimizer.state_dict(),
                "env_name": env.unwrapped.spec.id,
                "config": model_cfg
            }
            torch.save(checkpoint, savedir)


        """
        ============================================
                TEST DECISION TRANSFORMER                         
        ============================================
        """

        #wandb_log=wandb.init(project="Test DecisionTransformer")

        agent = DecisionTransformer(**model_cfg).to(device)
        if TRAIN==False:   
            file_name = f'state-{env.unwrapped.spec.id}.pt'
            path_weights = os.getcwd()+ f'\checkpoints\{file_name}'
            savedir = "state-hopper-bullet-medium-v0_5.pt" #Manual introduction of the weights (temporal)
        else:
            path_weights = savedir

        print("Weights will be loaded from: ",savedir)
        # Load the weights in the instance created Decision Transformer model
        checkpoint = torch.load(path_weights)

        agent.load_state_dict(checkpoint['model_state_dict'])
        print("Weights loaded succesfully")



        test_agent = TestAgent(agent,device,env,env_name)
        results = test_agent.test

        results = test_agent.test(context_len = hparams["context_len"],
                        rtg_target = hparams["rtg_target"], #max(val_rtgs),
                        rtg_scale = hparams["rtg_scale"],
                        constant_retrun_to_go = hparams["constant_retrun_to_go"],
                        num_test_ep = hparams["num_test_ep"],
                        max_test_ep_len = hparams["max_test_ep_len"],
                        state_mean = hparams["state_mean"], 
                        state_std = hparams["state_std"], 
                        render = hparams["render"]
                        )

        
       
        
        if env_id == len(env_names)-1:
            break  # Salir del bucle cuando se haya completado ultimo loop
        else:
            env_id = env_id + 1       


if __name__ == "__main__":
    main_loop()


print ("\n","/"*60)
print (" "*20,"Execution completed at:",datetime.datetime.now().strftime("%H:%MH%S"))
print ("/"*60)


