# Loading embedded trajectories in formats that work for imitation library
# Since imitation does not seem to support dictionaries easily, actions will be MultiDiscrete actions:
#   1. Button actions
#   2. Camera actions
#   3. ESC button
# 1. and 2. follow the original VPT model actiono space, while ESC button is new binary button for predicting
# when to press ESC (to end the episode)

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import glob

import numpy as np
from gymnasium import spaces
from imitation.data.types import Transitions
from tqdm import tqdm

from basalt.vpt_lib.agent import AGENT_NUM_BUTTON_ACTIONS, AGENT_NUM_CAMERA_ACTIONS

KEYS_FOR_TRANSITIONS = ["obs", "next_obs", "acts", "rewards", "dones", "infos"]

# Maximum number of transitions to load. Using a fixed array size avoids expensive recreation of arrays.
# This is hardcoded for `downsampling`=2.
# This is suitable for ~50GB of RAM.
MAX_DATA_SIZE = 1_000_000

def build_obs_and_act_gym_spaces(dataset):
    observation_space = spaces.Box(low=-float("inf"), high=float("inf"), shape=(dataset[0]["obs"].shape[1],))
    # 2 is for ESC button
    action_space = spaces.MultiDiscrete([AGENT_NUM_BUTTON_ACTIONS, AGENT_NUM_CAMERA_ACTIONS, 2])
    return observation_space, action_space

def get_all_npz_files_in_dir(dir_path):
    return glob.glob(os.path.join(dir_path, "*.npz"))

def hotfix_flattened_embeddings(embeddings, embedding_dim):
    """Reshape accidentally flattened embeddings back into correct shape"""
    return embeddings.reshape(-1, embedding_dim)

def load_embedded_trajectories_as_transitions(npz_file_paths, progress_bar=False, expected_embedding_dim=None, downsampling=1, skip_noops=False):

    episodes = []

    for npz_file_path in tqdm(npz_file_paths, desc="Loading trajectories", disable=not progress_bar, leave=False):
        # Create arrays with enough space which we then fill
        obs = np.zeros((MAX_DATA_SIZE, expected_embedding_dim), dtype=np.float32)
        next_obs = np.zeros((MAX_DATA_SIZE, expected_embedding_dim), dtype=np.float32)
        acts = np.zeros((MAX_DATA_SIZE, 3), dtype=np.int32)
        rewards = np.zeros((MAX_DATA_SIZE,), dtype=np.float32)
        dones = np.zeros((MAX_DATA_SIZE,), dtype=bool)
        infos = np.zeros((MAX_DATA_SIZE,), dtype=object)
        current_index = 0

        data = np.load(npz_file_path)
        try:
            embeddings = data["embeddings"]
            button_actions = data["button_actions"]
            camera_actions = data["camera_actions"]
            esc_actions = data["esc_actions"]
            is_null_action = data["is_null_action"]
        except KeyError as e:
            print(f"KeyError while loading {npz_file_path}: {e}")
            continue

        if embeddings.ndim == 1:
            assert expected_embedding_dim is not None, "Expected embedding dim must be provided if embeddings are flattened"
            embeddings = hotfix_flattened_embeddings(embeddings, expected_embedding_dim)

        if skip_noops:
            # Remove noops
            valid_action_mask = ~(is_null_action.astype(np.bool))
            embeddings = embeddings[valid_action_mask]
            button_actions = button_actions[valid_action_mask]
            camera_actions = camera_actions[valid_action_mask]
            esc_actions = esc_actions[valid_action_mask]

        # Downsampling
        embeddings = embeddings[::downsampling]
        button_actions = button_actions[::downsampling]
        camera_actions = camera_actions[::downsampling]
        esc_actions = esc_actions[::downsampling]

        assert embeddings.shape[0] == button_actions.shape[0] == camera_actions.shape[0] == esc_actions.shape[0], f"Shapes do not match: {embeddings.shape}, {button_actions.shape}, {camera_actions.shape}"

        # Add to arrays
        n = embeddings.shape[0]

        obs[current_index:current_index + n] = embeddings
        # Pad last observation with zeros
        next_obs[current_index:current_index + n] = np.concatenate((embeddings[1:], np.zeros((1, expected_embedding_dim))), axis=0)
        acts[current_index:current_index + n] = np.stack([button_actions, camera_actions, esc_actions], axis=1)
        rewards[current_index:current_index + n] = [0.0] * n
        rewards[current_index + n] = 10
        dones[current_index + n] = True
        current_index += n + 1

        if current_index >= MAX_DATA_SIZE:
            raise RuntimeError(f"Reached max data size of {MAX_DATA_SIZE} while loading trajectories. Increase `MAX_DATA_SIZE` entry or increase `downsampling` to load less data.")

        # Trim arrays to correct size
        obs = obs[:current_index]
        next_obs = next_obs[:current_index]
        acts = acts[:current_index]
        rewards = rewards[:current_index]
        dones = dones[:current_index]
        infos = infos[:current_index] 

        # EPV: Cut off arrays from the front
        start_index = max(len(obs) - 500, 0)
        obs = obs[start_index:]
        next_obs = next_obs[start_index:]
        acts = acts[start_index:]
        rewards = rewards[start_index:]
        dones = dones[start_index:]
        infos = infos[start_index:]
        
        # Create dictionary
        concat_all_parts = dict(zip(KEYS_FOR_TRANSITIONS, [obs, next_obs, acts, rewards, dones, infos]))

        episodes.append(concat_all_parts)

    return np.array(episodes)

def load_data_for_dt_from_path(data_path, expected_embedding_dim=None, max_files_to_load=None, downsampling=1, skip_noops=False):
    """
    Load data from a path in a format that can be used by the imitation library.
    Returns:
        transitions: imitation.data.types.Transitions
        observation_space: Observation space matching the observations
        action_space: Action space matching the actions
    """
    filelist = get_all_npz_files_in_dir(data_path)
    if max_files_to_load is not None:
        filelist = filelist[:max_files_to_load]
    transitions = load_embedded_trajectories_as_transitions(filelist, progress_bar=True, expected_embedding_dim=expected_embedding_dim, downsampling=downsampling, skip_noops=skip_noops)
    observation_space, action_space = build_obs_and_act_gym_spaces(transitions)
    return transitions, observation_space, action_space
 

class EpisodeDataset(Dataset):
    def __init__(self, data_path, expected_embedding_dim=None, max_files_to_load=None, downsampling=1, skip_noops=False):

        self.expected_embedding_dim = expected_embedding_dim
        self.downsampling = downsampling
        self.skip_noops = skip_noops

        self.episodes = get_all_npz_files_in_dir(data_path)
        if max_files_to_load is not None:
            self.episodes = self.episodes[:max_files_to_load]

    def __len__(self):
        return len(self.episodes)
    
    def get_state_and_act_dim(self):
        sample = self.__getitem__(0)

        state_dim = sample['obs'].shape[1] 
        act_dim = sample['acts'].shape[1] 
        observation_space, action_space = build_obs_and_act_gym_spaces([sample])
        return state_dim, act_dim, observation_space, action_space

    def __getitem__(self, idx):

        episode = self.episodes[idx]
        transitions = load_embedded_trajectories_as_transitions([episode], progress_bar=False, expected_embedding_dim=self.expected_embedding_dim, downsampling=self.downsampling, skip_noops=self.skip_noops)
        return transitions[0]
