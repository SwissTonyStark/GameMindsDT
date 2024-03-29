import datetime
import torch
import numpy as np
import wandb

from tqdm import tqdm
from utils import *

class TestAgent():
    def __init__(self,
        agent,
        device,
        env,
        env_name
        
    ):
        self.agent = agent
        self.device = device
        self.env = env
        self.env_name = env_name
        
    def test(self, 
             context_len, 
             rtg_target, 
             rtg_scale,
             constant_retrun_to_go,
             stochastic_start,
             num_test_ep, 
             max_test_ep_len,
             state_mean, 
             state_std, 
             render=False):

        agent_name = self.agent.__class__.__name__.lower()
        
        wandb_project_name = "Testing DT ["+ self.env_name + '] ['+datetime.datetime.now().strftime("%d_%m_%Y")+']'
        wandb_run_name = 'Test Run ['+datetime.datetime.now().strftime("%H.%M.%S")+']'
        wandb.init(project=wandb_project_name, name=wandb_run_name)
        
        test_parameters_log ={
            'context_len': context_len,
            'rtg_target': rtg_target,
            'rtg_scale': rtg_scale,
            'constant_retrun_to_go' : constant_retrun_to_go,
            'stochastic_start' : stochastic_start,
            'num_test_ep': num_test_ep,
            'max_test_ep_len': max_test_ep_len,
            'state_mean': state_mean,
            'state_std': state_std,
            'render': render,}
        
        wandb.config.update(test_parameters_log)

        
        # Define Weights&Biases special metric for evaluation episodes
        wandb.define_metric("num_test_episodes")
        wandb.log({"num_test_episodes": num_test_ep})

        # Define Weights&Biases special metric to draw using the evaluation episodes special metric
        wandb.define_metric("Average Return-to-go", step_metric="num_test_episodes")

        eval_batch_size = 1  # required for forward pass
        
        video_frames = []
        results = {}
        total_reward = 0
        total_returns_to_go =[]
        total_timesteps = 0
        

        state_dim = self.env.observation_space.shape[0]
        act_dim = self.env.action_space.shape[0]

        if state_mean is None:
            state_mean = torch.zeros((state_dim,)).to(self.device)
        else:
            state_mean = torch.Tensor(state_mean).to(self.device)

        if state_std is None:
            state_std = torch.ones((state_dim,)).to(self.device)
        else:
            state_std = torch.Tensor(state_std).to(self.device)

        # same as timesteps used for training the transformer
        # also, crashes if device is passed to arange()
        timesteps = torch.arange(start=0, end=max_test_ep_len, step=1)
        timesteps = timesteps.repeat(eval_batch_size, 1).to(self.device)

        self.agent.eval()

        # Initialize tqdm progess bar
        epoch_range = tqdm(range(num_test_ep), desc=f'Testing in Progress')

        # Potential & Return-to-go data windows
        window_potential = []
        window_return_to_go = []

        with torch.no_grad():

            for episode in range(num_test_ep):

                episode_returns = []
               
                # zeros place holders
                actions = torch.zeros((eval_batch_size, max_test_ep_len, act_dim),
                                    dtype=torch.float32, device=self.device)
                states = torch.zeros((eval_batch_size, max_test_ep_len, state_dim),
                                    dtype=torch.float32, device=self.device)
                rewards_to_go = torch.zeros((eval_batch_size, max_test_ep_len, 1),
                                    dtype=torch.float32, device=self.device)

                # Environment reset
                running_state = self.env.reset() #By using env_monitor.reset() we ensure that both the wrapper and the env are reseted synchronously
                
                # Add some noise to the initial observations, to prove generalization
                if stochastic_start:
                    running_state = running_state + np.random.normal(0, 0.1, size=running_state.shape)

                running_reward = 0
                running_rtg = rtg_target / rtg_scale

                timestep_range = tqdm(range(num_test_ep), desc=f'Test Episode [{episode+1}/{num_test_ep}] Running')

                for t in range(max_test_ep_len):

                    total_timesteps += 1

                    # Add state in placeholder and normalize
                    states[0, t] = torch.from_numpy(running_state).to(self.device)
                    states[0, t] = (states[0, t] - state_mean) / (state_std+1e-6)

                    # Calcualate running rtg and add it in placeholder
                    if constant_retrun_to_go:    
                            running_rtg = running_rtg
                    else:   
                            running_rtg = running_rtg - (running_reward / rtg_scale ) 

                    
                    rewards_to_go[0, t] = running_rtg 
                    

                    if t < context_len:
                        _, _, act_preds  = self.agent.forward(timesteps[:,:context_len],
                                                    states[:,:context_len],
                                                    actions[:,:context_len],
                                                    rewards_to_go[:,:context_len])
                        act = act_preds[0, t].detach()
                    else:
                        _, _, act_preds  = self.agent.forward(timesteps[:,t-context_len+1:t+1],
                                                    states[:,t-context_len+1:t+1],
                                                    actions[:,t-context_len+1:t+1],
                                                    rewards_to_go[:,t-context_len+1:t+1])
                        act = act_preds[0, -1].detach()
                         
                
                    
                    act = act.cpu().numpy()
              
                    running_state, running_reward, done, _ = self.env.step(act)
                    
                    video_frames.append(self.env.render(mode="rgb_array")) #mode="rgb_array" or mode="human"
                    
                    # Add action in placeholder (Actions Buffered)
                    
                    actions[0, t] = torch.cuda.FloatTensor(act)
                    #print("Tensor Act_Buffer", actions)

                    episode_returns.append(running_reward)
                    
                    total_returns_to_go.append(discounted_returns(episode_returns))
                    total_reward += running_reward
                    
                    wandb.log({"Episode Return-to-go": np.sum(episode_returns)}, step=total_timesteps)
                    
                    # Update Potential & Return_to_go data windows
                    window_potential.append(self.env.potential)
                    window_return_to_go.append(np.sum(episode_returns))

                    # Update Potential & Return_to_go data trend
                    trend_potential = trend_arrow(window_potential)
                    trend_return_to_go = trend_arrow(window_return_to_go)

                    # Update progress bar data
                    timestep_range.set_postfix({f'[{trend_potential}]Potential': self.env.potential,
                                                f'[{trend_return_to_go}]Episode Return-to-go': np.sum(episode_returns),
                                                f'Total Timesteps': total_timesteps})

                    # Actualiza la barra de progreso
                    timestep_range.update(1)
                    

                    if render:
                        self.env.render()
                    if done:
                        break
                    
                    #print(f'Episode [{episode}],  Timestep: [{t}/{max_test_ep_len}], Potential [{self.env.potential}],  Episode Return-to-go: {np.sum(episode_returns)}, Total Timesteps: {total_timesteps}')    
                    #print(f"Agent's action [{act[0]}],  Timestep: [{t}/{max_test_ep_len}], Episode Return-to-go: {np.sum(episode_returns)}, Total Timesteps: {total_timesteps}")      
                    #print(f'ObsDim1[{running_state[0]}], ObsDim2[{running_state[1]}], ObsDim3[{running_state[2]}], ObsDim4[{running_state[3]}], ObsDim5[{running_state[4]}], ObsDim6[{running_state[5]}], ObsDim7[{running_state[6]}], ObsDim8[{running_state[7]}], ObsDim9[{running_state[8]}], ObsDim10[{running_state[9]}], ObsDim11[{running_state[10]}], ObsDim12[{running_state[11]}], ObsDim13[{running_state[12]}], ObsDim14[{running_state[13]}], ObsDim15[{running_state[14]}]')
                
                timestep_range.close()
                epoch_range.update(1)

        results['Evaluation-avg_reward'] = total_reward / num_test_ep     
        results['Evaluation-avg_ep_len'] = total_timesteps / num_test_ep
        
        try:
            wandb.log({"Average Return-to-go": results['Evaluation-avg_reward']})#, "num_test_episodes": num_test_ep})
        except wandb.Error as e:
            print("Error during registering Data in wandb:", e)
            raise SystemExit(1)

        print(f"Average return-to-go over {num_test_ep} episodes: {results['Evaluation-avg_reward']}")
        

        #Video Management
        file_name = f'test_replay_{agent_name}_{self.env_name}' 

        video_recorded = generate_video_opencv(video_frames,file_name)
        
        try:
            wandb.log({"Video eval": wandb.Video(video_recorded, fps=4, format="")})
        except wandb.Error as e:
            print("Error during registering Video in wandb:", e)
            raise SystemExit(1)
        
        # Close tqdm progess bar
        epoch_range.close()

        # Close enviornment
        self.env.close()

        # Finish Weights and Biases run
        wandb.finish()

        return results




