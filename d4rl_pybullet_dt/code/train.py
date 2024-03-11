import torch
import wandb
import datetime
import math
import numpy as np

from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import trend_arrow, generate_video_opencv



class Trainer():
    def __init__(self, 
        model,
        env,
        env_name,  
        optimizer, 
        criterion, 
        device,
        hyperparameters
    ):
        self.model = model
        self.env = env
        self.env_name = env_name
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.hyperparameters = hyperparameters
    
    def __train_step(self, train_loader: DataLoader):
        train_loss = 0
        for states, actions, rtgs, steps, padd_mask in train_loader:

            states, actions, steps, rtgs, padd_mask = (x.to(self.device) for x in [states, actions, steps, rtgs, padd_mask])
            action_target = torch.clone(actions).detach().to(self.device)

            self.optimizer.zero_grad()

            _, _, act_preds = self.model.forward(steps, states, actions, rtgs) # timestep, max_timesteps, states, actions, returns_to_go

            act_preds = act_preds[padd_mask]
            action_target = action_target[padd_mask]
            loss = self.criterion(act_preds, action_target)
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

                _, _, act_preds = self.model.forward(steps, states, actions, rtgs)

                loss = self.criterion(act_preds, action_target)  
                val_loss += loss.item()
        return val_loss
    

    def __eval_env(self,eval_checkpoint,eval_period):

        model_name = self.model.__class__.__name__.lower()
        
        eval_batch_size = 1  # required for forward pass

        #Generate empty lists for the bests results in every epoch (Checkpoints every 10% of train epochs)
        if (0.1 * self.hyperparameters["train_epochs"]) % 1 == 0:

            best_video_buffer = [[] for _ in range(10)]
            best_acum_reward = [0 for _ in range(10)]
   
        else:
            best_video_buffer = [[] for _ in range(9)]
            best_acum_reward = [0 for _ in range(9)]
        

        state_dim = self.env.observation_space.shape[0]
        act_dim = self.env.action_space.shape[0]

        if self.hyperparameters['state_mean'] is None: 
            self.hyperparameters['state_mean'] = torch.zeros((state_dim,)).to(self.device)
        else:
            self.hyperparameters['state_mean'] = torch.Tensor(self.hyperparameters['state_mean']).to(self.device)

        if self.hyperparameters['state_std'] is None: 
            self.hyperparameters['state_std'] = torch.ones((state_dim,)).to(self.device)
        else:
            self.hyperparameters['state_std'] = torch.Tensor(self.hyperparameters['state_std']).to(self.device)

        # same as timesteps used for training the transformer
        # also, crashes if device is passed to arange()
        timesteps = torch.arange(start=0, end=self.hyperparameters['max_eval_ep_len'], step=1)
        timesteps = timesteps.repeat(eval_batch_size, 1).to(self.device)

        self.model.eval()

        # Initialize tqdm progess bar
        episode_range = tqdm(range(self.hyperparameters['num_eval_ep']), desc=f'Env.Evaluation Checkpoint {eval_checkpoint} Progress')


        with torch.no_grad():

            for episode in range(self.hyperparameters['num_eval_ep']): 

                video_frames = []

                total_reward = 0
                # zeros place holders
                actions = torch.zeros((eval_batch_size, self.hyperparameters['max_eval_ep_len'], act_dim),
                                    dtype=torch.float32, device=self.device)
                states = torch.zeros((eval_batch_size, self.hyperparameters['max_eval_ep_len'], state_dim),
                                    dtype=torch.float32, device=self.device)
                rewards_to_go = torch.zeros((eval_batch_size, self.hyperparameters['max_eval_ep_len'], 1),
                                    dtype=torch.float32, device=self.device)

                # Environment reset
                running_state = self.env.reset() #By using env_monitor.reset() we ensure that both the wrapper and the env are reseted synchronously

                # Add some noise to the initial observations, to prove generalization
                if self.hyperparameters['stochastic_start']:
                    running_state = running_state + np.random.normal(0, 0.1, size=running_state.shape)

                running_reward = 0
                running_rtg = self.hyperparameters['rtg_target'] / self.hyperparameters['rtg_scale']  
                

                for t in range(self.hyperparameters['max_eval_ep_len']):

                    # Add state in placeholder and normalize
                    states[0, t] = torch.from_numpy(running_state).to(self.device)
                    states[0, t] = (states[0, t] - self.hyperparameters['state_mean']) / (self.hyperparameters['state_std']+1e-6)

                    # Calcualate running rtg and add it in placeholder
                    if self.hyperparameters['constant_retrun_to_go']:    
                            running_rtg = running_rtg
                    else:   
                            running_rtg = running_rtg - (running_reward / self.hyperparameters['rtg_scale'] ) 
                    
                    rewards_to_go[0, t] = running_rtg 
                    

                    if t < self.hyperparameters['context_len']:
                        _, _, act_preds  = self.model.forward(timesteps[:,:self.hyperparameters['context_len']],
                                                    states[:,:self.hyperparameters['context_len']],
                                                    actions[:,:self.hyperparameters['context_len']],
                                                    rewards_to_go[:,:self.hyperparameters['context_len']])
                        act = act_preds[0, t].detach() 
                    else:
                        _, _, act_preds  = self.model.forward(timesteps[:,t-self.hyperparameters['context_len']+1:t+1],
                                                    states[:,t-self.hyperparameters['context_len']+1:t+1],
                                                    actions[:,t-self.hyperparameters['context_len']+1:t+1],
                                                    rewards_to_go[:,t-self.hyperparameters['context_len']+1:t+1])
                        act = act_preds[0, -1].detach()
                         
                    act = act.cpu().numpy()

                    running_state, running_reward, done, _ = self.env.step(act)
                    
                    video_frames.append(self.env.render(mode="rgb_array"))

                    # Add action in placeholder (Actions Buffered)
                    actions[0, t] = torch.cuda.FloatTensor(act)
                    
                    total_reward += running_reward

                    # Update progress bar data
                    episode_range.set_postfix({})

                    # Actualiza la barra de progreso
                    episode_range.update(1)
                    
                    # Register best frames and total reward for every eval_checkpoint
                    if  total_reward > best_acum_reward[eval_checkpoint-1]:
                            best_acum_reward[eval_checkpoint-1] = total_reward
                            best_video_buffer[eval_checkpoint-1] = video_frames[:]

                    if done:
                        if total_reward > best_acum_reward[eval_checkpoint-1]:
                            best_acum_reward[eval_checkpoint-1] = total_reward
                            best_video_buffer[eval_checkpoint-1] = video_frames[:]
                        break   
        
        # Close tqdm progess bar
        episode_range.close()

        #Video Management           
        file_name = f'train_checkpoint{eval_checkpoint}_{model_name}_{self.env_name}' 
        
        video_recorded = generate_video_opencv(best_video_buffer[eval_checkpoint-1],file_name)

        wandb.log({f'Train Checkpoint Nº{eval_checkpoint}': wandb.Video(video_recorded, fps=4, format="")})

        return best_acum_reward 



    def train(self, train_loader: DataLoader, val_loader: DataLoader):

        
        timestamp_project = datetime.datetime.now().strftime("%d_%m_%Y") 
        
        # WandB initialization
        wandb_project_name = "Training DT ["+ self.env_name + '] ['+timestamp_project+']'
        wandb_run_name = 'Train Run ['+datetime.datetime.now().strftime("%H.%M.%S")+']'
        wandb.init(project=wandb_project_name, name=wandb_run_name)

        # Track gradients (Tracks the Optimizer, Criterion and other Pytorch functions recognized)
        wandb.watch(self.model)
        
        # Track hyperparameters used
        hyperparameters_log ={
            'h_dim': self.hyperparameters['h_dim'],
            'num_heads': self.hyperparameters['num_heads'],
            'num_blocks': self.hyperparameters['num_blocks'],
            'context_len': self.hyperparameters['context_len'],
            'batch_size': self.hyperparameters['batch_size'],
            'lr': self.hyperparameters['lr'],
            'weight_decay': self.hyperparameters['weight_decay'],
            'mlp_ratio': self.hyperparameters['mlp_ratio'],
            'dropout': self.hyperparameters['dropout'],
            'train_epochs': self.hyperparameters['train_epochs'],
            'rtg_target': self.hyperparameters['rtg_target'],
            'rtg_scale': self.hyperparameters['rtg_scale'],
            'constant_retrun_to_go': self.hyperparameters['constant_retrun_to_go'],
            'stochastic_start' : self.hyperparameters['stochastic_start'],
            'num_eval_ep': self.hyperparameters['num_eval_ep'],
            'max_eval_ep_len': self.hyperparameters['max_eval_ep_len'],
            'state_mean': self.hyperparameters['state_mean'],
            'state_std': self.hyperparameters['state_std'],
            }
                
        wandb.config.update(hyperparameters_log)

        # Track gradients (Tracks the Optimizer, Criterion and other Pytorch functions recognized)
        wandb.watch(self.model)

        #Initialize Environment Evaluation Checkpoints
        eval_checkpoint = 1
        eval_period = math.floor((0.1) * self.hyperparameters["train_epochs"]) #10% of the training epochs
        next_eval_checkpoint = eval_period

        # Initialize tqdm progess bar
        epoch_range = tqdm(range(self.hyperparameters['train_epochs']), desc=f'Training in Progress')

        # Train & Validation data losses' windows
        window_train_losses = []
        window_val_losses = []

        for epoch in range(self.hyperparameters["train_epochs"]):

            # Training Step
            self.model.train()
            train_loss = self.__train_step(train_loader)

            # Evaluation Step
            self.model.eval()
            val_loss = self.__val_step(val_loader)

            #Evaluation Environment
            if epoch == next_eval_checkpoint:
                #todo 
                print(f'\n** Envionment Evaluation Checkpoint Nº{eval_checkpoint} STARTED **')
                best_acum_reward = self.__eval_env(eval_checkpoint,eval_period)  
                print(f'Best Acumulated Reward in checkpoint Nº{eval_checkpoint}: {best_acum_reward[eval_checkpoint-1]}\n')

                eval_checkpoint = eval_checkpoint + 1
                next_eval_checkpoint = next_eval_checkpoint + eval_period


            epoch_train_loss = train_loss / len(train_loader.dataset)
            epoch_val_loss = val_loss / len(val_loader.dataset) 

            # Update Train & Validation data losses' windows
            window_train_losses.append(epoch_train_loss)
            window_val_losses.append(epoch_val_loss)

            # Update Train & Validation data losses trend
            trend_train_loss = trend_arrow(window_train_losses)
            trend_val_loss = trend_arrow(window_val_losses)
            
            try:
                wandb.log({'Training Loss Average': epoch_train_loss, 'Validation Loss Average': epoch_val_loss})
            except wandb.Error as e:
                print("Error during registering Data in wandb:", e)
                raise SystemExit(1)
            
            # Update progress bar data
            epoch_range.set_postfix({f"[{trend_train_loss}]Training Loss Average": epoch_train_loss, f"[{trend_val_loss}]Validation Loss Average": epoch_val_loss})

            # Update ratio progress bar 
            epoch_range.update(1)

            # Update epoch, training and validation info
            #print(f'Epoch [{epoch+1}/{self.hyperparameters["train_epochs"]}], Training Loss Average: {epoch_train_loss:.10f}, Validation Loss Average: {epoch_val_loss:.10f}')

        # Close tqdm progess bar
        epoch_range.close()   

        # Finish Weights and Biases run
        wandb.finish()


       
