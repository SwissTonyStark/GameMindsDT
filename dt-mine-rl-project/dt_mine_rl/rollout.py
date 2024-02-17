import argparse
import os

import torch
import numpy as np
import tqdm
import pprint

from imitation.algorithms import bc
import videoio

from vpt_lib.agent import MineRLAgent

from config import config
from lib.common import ENV_NAME_TO_SPEC, ENV_TO_BASALT_2022_SEEDS, create_json_entry_dict, new_create_observables, load_model_parameters

from dt_models.dt_model_common import ActEncoderDecoder
from dt_models.dt_model_hf import TrainableDT


ENV_TARGETS = {
    "MineRLBasaltFindCave-v0": [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
    "MineRLBasaltMakeWaterfall-v0": [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
}

ENV_TEMPERATURES_CAMERA = {
    "MineRLBasaltFindCave-v0": [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
    "MineRLBasaltMakeWaterfall-v0": [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
}

#BasaltBaseEnvSpec.create_observables = new_create_observables

def main(args):

    env_key = "MineRLBasaltFindCave-v0"
    app_config = config["envs"][env_key]
    
    mode = app_config["mode"]

    act_button_encoder = None

    if (app_config["button_act_csv_path"] is not None):
        act_button_encoder = ActEncoderDecoder(app_config["button_act_csv_path"], app_config["button_encoder_num_actions"])
    
    scale = app_config["scale_rewards"]

    device = app_config["rollout_device"]

    vpt_agent_policy_kwargs, vpt_agent_pi_head_kwargs = load_model_parameters(app_config["vpt_model"])

    vpt_agent = MineRLAgent(policy_kwargs=vpt_agent_policy_kwargs, pi_head_kwargs=vpt_agent_pi_head_kwargs)
    vpt_agent.load_weights(app_config["vpt_weights"])
    vpt_agent.policy.eval()

    dummy_first = torch.from_numpy(np.array((False,))).cuda().to(device)

    agent = TrainableDT.from_pretrained(app_config["models_dir"])
    agent.set_default_temperatures(app_config["temperature_buttons"], app_config["temperature_camera"], app_config["temperature_esc"])
    agent.set_disable_esc_button()

    pprint.PrettyPrinter(indent=4,sort_dicts=True).pprint(agent)

    agent.to(device)
    
    env = ENV_NAME_TO_SPEC[env_key]().make()

    # Patch so that we get more statistics for tracking purposes
    env.create_observables = new_create_observables

    environment_seeds = app_config["environment_seeds"]
    if environment_seeds is None:
        environment_seeds = ENV_TO_BASALT_2022_SEEDS[env_key]


    os.makedirs(app_config["rollout_output_dir"], exist_ok=True)

    # Reset env extra time to ensure that the first setting will work fine
    env.reset()

    idx = 0
    for seed in tqdm.tqdm(environment_seeds, desc="Seeds", leave=False):
        TARGET_RETURN = ENV_TARGETS[env_key][idx]

        env.seed(seed)
        obs = env.reset()
        hidden_state = vpt_agent.policy.initial_state(1)
        recorder = videoio.VideoWriter(os.path.join(app_config["rollout_output_dir"], f"seed_{seed}_{idx}.mp4"), resolution=(640, 360), fps=20)

        target_return = torch.tensor(TARGET_RETURN, device=device, dtype=torch.float32).reshape(1, 1)
        
        states = None
        actions = torch.zeros((0, agent.config.act_dim), device=device, dtype=torch.float32)
        rewards = torch.zeros(0, device=device, dtype=torch.float32)

        timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)

        json_data = []
        recorder.write(obs["pov"])

        done = False
        progress_bar = tqdm.tqdm(desc=f"Steps", leave=False)
        step_counter = 0

        while not done:

            # The agent is only allowed to see the "pov" entry of the observation.
            # This function takes the "pov" observation and resizes it for the agent.
            agent_obs = vpt_agent._env_obs_to_agent(obs)
            with torch.no_grad():

                vpt_embedding, hidden_state = vpt_agent.policy.get_output_for_observation(
                    agent_obs,
                    hidden_state,
                    dummy_first,
                    return_embedding=True,
                )

                cur_state = vpt_embedding.to(device=device).reshape(1, agent.config.state_dim)
                if (states is None):
                    states = cur_state
                else:
                    states = torch.cat([states, cur_state], dim=0)

                agent_action = agent.get_dt_action(
                    states,
                    actions,
                    rewards,
                    target_return,
                    timesteps,
                    device,
                    temperature_camera=ENV_TEMPERATURES_CAMERA[env_key][idx]
                )
                actions = torch.cat([actions, torch.zeros((1, agent.config.act_dim), device=device)], dim=0)
                rewards = torch.cat([rewards, torch.zeros(1, device=device)])
                actions[-1] = agent_action[-1]

    
            # We need to have both batch and seq dimensions for the actions
            
            button_index = agent_action[:, 0].cpu().numpy()[0]
            
            agent_action_dict = {
                "buttons": torch.tensor([act_button_encoder.decode(button_index)], device=device, dtype=torch.long).unsqueeze(0),
                "camera": agent_action[:, 1].unsqueeze(0)
            }

            minerl_action = vpt_agent._agent_action_to_env(agent_action_dict)

            minerl_action["ESC"] = agent_action[0, 2].cpu().numpy()

            # Add the symbolic data here so that video and json are in sync
            #json_data.append(create_json_entry_dict(obs, minerl_action))

            obs, _, done, _ = env.step(minerl_action)

            if mode != 'delayed':
                #reward = dt_critic_model.forward(vpt_embedding.cpu())

                #reward = th.where(reward < 0.9, th.zeros_like(reward), reward * 1)  
                
                #reward = th.tensor([0]).to(device)

                #if reward > 0.9:
                #    print("reward", reward)
            
                reward = torch.tensor([1]).to(device)
                #rewards[-1] = reward                    
                pred_return = target_return[0, -1] - (reward / scale)
            else:
                pred_return = target_return[0, -1]

            #print(pred_return)
            target_return = torch.cat([target_return, pred_return.reshape(1, 1)], dim=1)

            #print(target_return.shape)
            #timesteps = th.cat([timesteps, th.ones((1, 1), device=device, dtype=th.long) * (0)], dim=1)
            timesteps = torch.cat([timesteps, torch.ones((1, 1), device=device, dtype=torch.long) * step_counter + 1], dim=1)
            #timesteps = th.cat([timesteps, th.ones((1, 1), device=device, dtype=th.long) * ((step_counter + 1)% max_ep_len)], dim=1)
            #timesteps = th.arange(target_return.size(1), device=target_return.device)

            recorder.write(obs["pov"])
            progress_bar.update(1)
            step_counter += 1
            if app_config["rollout_max_steeps_per_seed"] is not None and step_counter >= app_config["rollout_max_steeps_per_seed"]:
                break
        recorder.close()

        # Write the jsonl file
        #with open(os.path.join(app_config["rollout_output_dir"], f"seed_{seed}_{idx}.jsonl"), "w") as f:
        #    f.write("\n".join(json_data))
        idx += 1
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main(args)
