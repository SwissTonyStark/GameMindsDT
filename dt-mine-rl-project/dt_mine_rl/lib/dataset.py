
from .common import KEYS_FOR_TRANSITIONS, MAX_DATA_SIZE
from torch.utils.data import Dataset
import numpy as np
import glob
from tqdm import tqdm
import os


class EpisodeDataset(Dataset):
    def __init__(self, data_path, expected_embedding_dim=None, max_files_to_load=None, downsampling=1, skip_noops=False, act_button_encoder=None , end_cut_episode_length=None, end_episode_margin=0):

        self.expected_embedding_dim = expected_embedding_dim
        self.downsampling = downsampling
        self.skip_noops = skip_noops

        self.act_button_encoder = act_button_encoder

        self.episodes = get_all_npz_files_in_dir(data_path)
        if max_files_to_load is not None:
            self.episodes = self.episodes[:max_files_to_load]

        self.end_cut_episode_length = end_cut_episode_length
        self.end_episode_margin = end_episode_margin

    def __len__(self):
        return len(self.episodes)
    
    def get_state_and_act_dim(self):
        sample = self.__getitem__(0)

        state_dim = sample['obs'].shape[1] 
        act_dim = sample['acts'].shape[1] 

        return state_dim, act_dim
    
    def __getitem__(self, idx):

        episode = self.episodes[idx]
        transitions = load_embedded_trajectories_as_transitions(
            [episode], self.act_button_encoder, progress_bar=False, expected_embedding_dim=self.expected_embedding_dim, 
            downsampling=self.downsampling, skip_noops=self.skip_noops, end_cut_episode_length=self.end_cut_episode_length, end_episode_margin=self.end_episode_margin)
        return transitions[0]


def get_all_npz_files_in_dir(dir_path):
    return glob.glob(os.path.join(dir_path, "*.npz"))

def hotfix_flattened_embeddings(embeddings, embedding_dim):
    """Reshape accidentally flattened embeddings back into correct shape"""
    return embeddings.reshape(-1, embedding_dim)

def load_embedded_trajectories_as_transitions(npz_file_paths, act_button_encoder=None, 
                                              max_episode_length=None, progress_bar=False, expected_embedding_dim=None, downsampling=1, 
                                              skip_noops=False, end_cut_episode_length=None, end_episode_margin=0):

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

        if act_button_encoder is not None:
            button_actions = np.array([act_button_encoder.encode(act) for act in button_actions])
            button_mask = np.array([False if action is None else True for action in button_actions])

            embeddings = embeddings[button_mask]
            button_actions = button_actions[button_mask]
            camera_actions = camera_actions[button_mask]
            esc_actions = esc_actions[button_mask]
            is_null_action = is_null_action[button_mask]

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

        rewards[current_index + n - end_episode_margin] = 100.0
        dones[current_index + n - end_episode_margin] = True
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


        if end_cut_episode_length is not None:
            if (end_episode_margin is None):
                end_episode_margin = 0
            if (len(obs)>(end_cut_episode_length + end_episode_margin)):
                obs = obs[-(end_cut_episode_length + end_episode_margin): -end_episode_margin]
                next_obs = next_obs[-(end_cut_episode_length + end_episode_margin):-end_episode_margin]
                acts = acts[-(end_cut_episode_length + end_episode_margin):-end_episode_margin]
                rewards = rewards[-(end_cut_episode_length + end_episode_margin):-end_episode_margin]
                dones = dones[-(end_cut_episode_length + end_episode_margin):-end_episode_margin]
                infos = infos[-(end_cut_episode_length + end_episode_margin):-end_episode_margin]


        # Create dictionary
        concat_all_parts = dict(zip(KEYS_FOR_TRANSITIONS, [obs, next_obs, acts, rewards, dones, infos]))

        episodes.append(concat_all_parts)


    return np.array(episodes)


 