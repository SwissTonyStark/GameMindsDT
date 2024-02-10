import argparse
import os
import json


import numpy as np
import torch as th
import gym
import minerl
import tqdm
from minerl.herobraine.env_specs.basalt_specs import BasaltBaseEnvSpec, FindCaveEnvSpec, MakeWaterfallEnvSpec, PenAnimalsVillageEnvSpec, VillageMakeHouseEnvSpec
from minerl.herobraine.hero.mc import ALL_ITEMS
from minerl.herobraine.hero import handlers
from imitation.algorithms import bc
import videoio

from basalt.embed_trajectories import load_model_parameters
from basalt.vpt_lib.agent import MineRLAgent

from basalt.dt_model import TrainableDT

from basalt.vpt_lib.agent import AGENT_NUM_BUTTON_ACTIONS, AGENT_NUM_CAMERA_ACTIONS


IMAGE_RESOLUTION = (640, 360)
# Manual mapping so that below patching works
ENV_NAME_TO_SPEC = {
    "MineRLBasaltFindCave-v0": FindCaveEnvSpec,
    "MineRLBasaltMakeWaterfall-v0": MakeWaterfallEnvSpec,
    "MineRLBasaltCreateVillageAnimalPen-v0": PenAnimalsVillageEnvSpec,
    "MineRLBasaltBuildVillageHouse-v0": VillageMakeHouseEnvSpec,
}

# These are the seeds per environment models were evaluated on
ENV_TO_BASALT_2022_SEEDS = {
    #"MineRLBasaltFindCave-v0": [14169, 65101, 78472, 76379, 39802, 95099, 63686, 49077, 77533, 31703, 73365],
    "MineRLBasaltMakeWaterfall-v0": [95674, 39036, 70373, 84685, 91255, 56595, 53737, 12095, 86455, 19570, 40250],
    "MineRLBasaltCreateVillageAnimalPen-v0": [21212, 85236, 14975, 57764, 56029, 65215, 83805, 35884, 27406, 5681265, 20848],
    "MineRLBasaltBuildVillageHouse-v0": [52216, 29342, 67640, 73169, 86898, 70333, 12658, 99066, 92974, 32150, 78702],
    "MineRLBasaltFindCave-v0": [65101, 65101, 65101, 65101, 65101, 65101],
}

ENV_TARGETS = {
    "MineRLBasaltFindCave-v0": [10, 10, 10, 10, 10, 10]
}

ENV_TEMPERATURES_CAMERA = {
    "MineRLBasaltFindCave-v0": [1.81, 1.81, 1.81, 1.81, 1.81, 1.81 ]
}

KEYS_OF_INTEREST = ['equipped_items', 'life_stats', 'location_stats', 'use_item', 'drop', 'pickup', 'break_item', 'craft_item', 'mine_block', 'damage_dealt', 'entity_killed_by', 'kill_entity', 'full_stats']

# Hotpatch the MineRL BASALT envs to report more statistics.
# NOTE: this is only to get more information on what agent is doing.
#       None of this information is passed to the agent.
def new_create_observables(self):
    obs_handler_pov = handlers.POVObservation(self.resolution)
    return [
        obs_handler_pov,
        handlers.EquippedItemObservation(
            items=ALL_ITEMS,
            mainhand=True,
            offhand=True,
            armor=True,
            _default="air",
            _other="air",
        ),
        handlers.ObservationFromLifeStats(),
        handlers.ObservationFromCurrentLocation(),
        handlers.ObserveFromFullStats("use_item"),
        handlers.ObserveFromFullStats("drop"),
        handlers.ObserveFromFullStats("pickup"),
        handlers.ObserveFromFullStats("break_item"),
        handlers.ObserveFromFullStats("craft_item"),
        handlers.ObserveFromFullStats("mine_block"),
        handlers.ObserveFromFullStats("damage_dealt"),
        handlers.ObserveFromFullStats("entity_killed_by"),
        handlers.ObserveFromFullStats("kill_entity"),
        handlers.ObserveFromFullStats(None),
    ]
BasaltBaseEnvSpec.create_observables = new_create_observables

def add_rollout_specific_args(parser):
    parser.add_argument("--output_dir", type=str, required=True, help="Where to store the rollout results")

    parser.add_argument("--agent_type", type=str, default="bc", choices=["bc","dt"], help="Model type.")
    parser.add_argument("--agent_file", type=str, required=True, help="Path to the trained model to rollout")

    parser.add_argument("--vpt_model", required=True, type=str, help="Path to the .model file to be used for embedding")
    parser.add_argument("--vpt_weights", required=True, type=str, help="Path to the .weights file to be used for embedding")

    parser.add_argument("--env", required=True, type=str, choices=ENV_NAME_TO_SPEC.keys(), help="Name of the environment to roll agent in")
    parser.add_argument("--environment_seeds", default=None, nargs="+", type=int, help="Environment seeds to roll out on, one per video.")

    parser.add_argument("--max_steps_per_seed", default=500, type=int, help="Maximum number of steps to run for per seed")

def remove_numpyness_and_remove_zeros(dict_with_numpy_arrays):
    # Recursively remove numpyness from a dictionary.
    # Remove zeros from the dictionary as well to save space.
    if isinstance(dict_with_numpy_arrays, dict):
        new_dict = {}
        for key, value in dict_with_numpy_arrays.items():
            new_value = remove_numpyness_and_remove_zeros(value)
            if new_value != 0:
                new_dict[key] = new_value
        return new_dict
    elif isinstance(dict_with_numpy_arrays, np.ndarray):
        if dict_with_numpy_arrays.size == 1:
            return dict_with_numpy_arrays.item()
        else:
            return dict_with_numpy_arrays.tolist()

def create_json_entry_dict(obs, action):
    stats = {}
    for key in KEYS_OF_INTEREST:
        stats[key] = remove_numpyness_and_remove_zeros(obs[key])
    stats["action"] = remove_numpyness_and_remove_zeros(action)
    stats = json.dumps(stats)
    return stats


def get_dt_action(model, states, actions, rewards, returns_to_go, timesteps, device, temperature_camera=0.5):
    # This implementation does not condition on past rewards

    states = states.reshape(1, -1, model.config.state_dim)
    actions = actions.reshape(1, -1, model.config.act_dim)
    returns_to_go = returns_to_go.reshape(1, -1, 1)
    timesteps = timesteps.reshape(1, -1)

    states = states[:, -model.config.max_length :]
    actions = actions[:, -model.config.max_length :]
    returns_to_go = returns_to_go[:, -model.config.max_length :]
    timesteps = timesteps[:, -model.config.max_length :]
    padding = model.config.max_length - states.shape[1]
    # pad all tokens to sequence length
    attention_mask = th.cat([th.zeros(padding, device=device), th.ones(states.shape[1], device=device)])
    attention_mask = attention_mask.to(dtype=th.long).reshape(1, -1)
    states = th.cat([th.zeros((1, padding, model.config.state_dim), device=device), states], dim=1).float()
    actions = th.cat([th.zeros((1, padding, model.config.act_dim), device=device), actions], dim=1).float()
    returns_to_go = th.cat([th.zeros((1, padding, 1), device=device), returns_to_go], dim=1).float()
    #print("returns_to_go.shape", returns_to_go.shape)
    timesteps = th.cat([th.zeros((1, padding), device=device, dtype=th.long), timesteps], dim=1)
    #print("timesteps.shape", timesteps.shape)
    
    state_preds, action_logits, return_preds = model.original_forward(
        states=states,
        actions=actions,
        rewards=rewards,
        returns_to_go=returns_to_go,
        timesteps=timesteps,
        attention_mask=attention_mask,
        return_dict=False,
    )

    temperature = 1
    #temperature_button = 1.25 * AGENT_NUM_BUTTON_ACTIONS / (AGENT_NUM_BUTTON_ACTIONS + AGENT_NUM_CAMERA_ACTIONS + 2)
    #temperature_camera = 0.75 * AGENT_NUM_CAMERA_ACTIONS / (AGENT_NUM_BUTTON_ACTIONS + AGENT_NUM_CAMERA_ACTIONS + 2)
    #temperature_esc = temperature * 2 / (AGENT_NUM_BUTTON_ACTIONS + AGENT_NUM_CAMERA_ACTIONS + 2)
    temperature_button = 1 #1.1 Best
    temperature_camera = temperature_camera #0.45 Best
    temperature_esc = 1


    agent_num_button_actions = AGENT_NUM_BUTTON_ACTIONS
    agent_num_camera_actions = AGENT_NUM_CAMERA_ACTIONS
    agent_esc_button = 2

    action_logits_button = action_logits[:,-1,:agent_num_button_actions]
    action_logits_camera = action_logits[:,-1,agent_num_button_actions:agent_num_button_actions + agent_num_camera_actions]
    action_logits_esc = action_logits[:,-1,agent_num_button_actions + agent_num_camera_actions:]

    action_logits_button = action_logits_button / temperature_button
    action_logits_camera = action_logits_camera / temperature_camera
    action_logits_esc = action_logits_esc / temperature_esc

    action_probs_button = th.softmax(action_logits_button, dim=-1)
    action_probs_camera = th.softmax(action_logits_camera, dim=-1)
    action_probs_esc = th.softmax(action_logits_esc, dim=-1)

    action_preds_button = th.multinomial(action_probs_button, num_samples=1).squeeze(-1)
    action_preds_camera = th.multinomial(action_probs_camera, num_samples=1).squeeze(-1)

    action_preds_esc = th.multinomial(action_probs_esc, num_samples=1).squeeze(-1)
    action_preds_esc = th.zeros(1, device=device, dtype=th.long)

    action_preds = th.stack([action_preds_button, action_preds_camera, action_preds_esc], dim=-1)

    return action_preds

def main(args):

    TARGET_RETURN = 1000.0

    scale = 1
    #device = th.device('cuda' if th.cuda.is_available() else 'cpu')
    device = "cpu"

    vpt_agent_policy_kwargs, vpt_agent_pi_head_kwargs = load_model_parameters(args.vpt_model)

    vpt_agent = MineRLAgent(policy_kwargs=vpt_agent_policy_kwargs, pi_head_kwargs=vpt_agent_pi_head_kwargs)
    vpt_agent.load_weights(args.vpt_weights)
    vpt_agent.policy.eval()

    #dt_critic_model = SimpleBinaryClassifier(1024)

    #dt_critic_model.load_state_dict(th.load(args.agent_file + '/model_dt_critic.pth', map_location=th.device(device)))
    #dt_critic_model.eval()

    dummy_first = th.from_numpy(np.array((False,))).cuda().to(device)

    agent = None
    if args.agent_type == "bc":
        agent = bc.reconstruct_policy(args.agent_file)
    elif args.agent_type == "dt":
        agent = TrainableDT.from_pretrained(args.agent_file)
    print("Agent config", agent.config)
    agent.to(device)
    env = ENV_NAME_TO_SPEC[args.env]().make()
    # Patch so that we get more statistics for tracking purposes
    env.create_observables = new_create_observables

    environment_seeds = args.environment_seeds
    if environment_seeds is None:
        environment_seeds = ENV_TO_BASALT_2022_SEEDS[args.env]

    os.makedirs(args.output_dir, exist_ok=True)

    # Reset env extra time to ensure that the first setting will work fine
    env.reset()

    state_dim = 1024
    act_dim = 3

    idx = 0
    for seed in tqdm.tqdm(environment_seeds, desc="Seeds", leave=False):
        TARGET_RETURN = ENV_TARGETS[args.env][idx]

        env.seed(seed)
        obs = env.reset()
        hidden_state = vpt_agent.policy.initial_state(1)
        recorder = videoio.VideoWriter(os.path.join(args.output_dir, f"seed_{seed}_{idx}.mp4"), resolution=(640, 360), fps=20)

        target_return = th.tensor(TARGET_RETURN, device=device, dtype=th.float32).reshape(1, 1)
        
        #target_return = th.arange(-(agent.config.max_length - 1), 1, dtype=th.float32)
        #target_return = target_return.flip(dims=[0]).unsqueeze(0)
        
        print("target_return", target_return.shape)
        
        states = None
        actions = th.zeros((0, act_dim), device=device, dtype=th.float32)
        rewards = th.zeros(0, device=device, dtype=th.float32)

        timesteps = th.tensor(0, device=device, dtype=th.long).reshape(1, 1)

        json_data = []
        recorder.write(obs["pov"])

        done = False
        progress_bar = tqdm.tqdm(desc=f"Steps", leave=False)
        step_counter = 0
        while not done:
            # The agent is only allowed to see the "pov" entry of the observation.
            # This function takes the "pov" observation and resizes it for the agent.
            agent_obs = vpt_agent._env_obs_to_agent(obs)
            with th.no_grad():
                vpt_embedding, hidden_state = vpt_agent.policy.get_output_for_observation(
                    agent_obs,
                    hidden_state,
                    dummy_first,
                    return_embedding=True,
                )
                if (args.agent_type == "dt"):
                    actions = th.cat([actions, th.zeros((1, act_dim), device=device)], dim=0)
                    rewards = th.cat([rewards, th.zeros(1, device=device)])
                    cur_state = vpt_embedding.to(device=device).reshape(1, state_dim)
                    if (states is None):
                        states = cur_state
                    else:
                        states = th.cat([states, cur_state], dim=0)

                    agent_action = get_dt_action(
                        agent,
                        states,
                        actions,
                        rewards,
                        target_return,
                        timesteps,
                        device,
                        temperature_camera=ENV_TEMPERATURES_CAMERA[args.env][idx]
                    )
                    actions[-1] = agent_action[0, -1]
                else:
                    agent_action, _, _ = agent(vpt_embedding[0])
            
            # We need to have both batch and seq dimensions for the actions

            agent_action_dict = {
                "buttons": agent_action[:, 0].unsqueeze(0),
                "camera": agent_action[:, 1].unsqueeze(0)
            }

            minerl_action = vpt_agent._agent_action_to_env(agent_action_dict)

            minerl_action["ESC"] = agent_action[0, 2].cpu().numpy()

            # Add the symbolic data here so that video and json are in sync
            json_data.append(create_json_entry_dict(obs, minerl_action))

            obs, _, done, _ = env.step(minerl_action)

            if (args.agent_type == "dt"):
                #reward = dt_critic_model.forward(vpt_embedding.cpu())

                #reward = th.where(reward < 0.85, th.zeros_like(reward), reward)
            
                reward = th.tensor([0.01]).to(device)
                rewards[-1] = reward
                
                pred_return = target_return[0, -1] - (reward / scale)
                #print(pred_return)
                target_return = th.cat([target_return, pred_return.reshape(1, 1)], dim=1)
                #print(target_return.shape)
                timesteps = th.cat([timesteps, th.ones((1, 1), device=device, dtype=th.long) * (step_counter + 1)], dim=1)
                #timesteps = th.cat([timesteps, th.ones((1, 1), device=device, dtype=th.long) * ((step_counter + 1)% max_ep_len)], dim=1)
                #timesteps = th.arange(target_return.size(1), device=target_return.device)


            recorder.write(obs["pov"])
            progress_bar.update(1)
            step_counter += 1
            if args.max_steps_per_seed is not None and step_counter >= args.max_steps_per_seed:
                break
        recorder.close()

        # Write the jsonl file
        with open(os.path.join(args.output_dir, f"seed_{seed}_{idx}.jsonl"), "w") as f:
            f.write("\n".join(json_data))
        idx += 1
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_rollout_specific_args(parser)
    args = parser.parse_args()
    main(args)
