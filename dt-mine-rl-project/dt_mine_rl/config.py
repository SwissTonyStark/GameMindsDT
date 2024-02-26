import os
from dt_mine_rl.lib.common import ENV_TO_BASALT_2022_SEEDS
import yaml
import torch

# Load settings.yaml
yaml_path = os.path.join(os.path.dirname(__file__), "..", "settings.yaml")

if not os.path.exists(yaml_path):
    raise FileNotFoundError(f"settings.yaml not found at {yaml_path}, please create it. See settings.yaml.example for an example.")

with open(yaml_path, "r") as f:
    settings = yaml.safe_load(f)
path_data = settings["path_data"]

cuda_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("Settings", settings)

config = {
    "common": {
        "reduce_button_actions": True,
        "path_data": path_data,
        "skip_if_exists": False,
        "vpt_model": "foundation-model-1x.model",
        "vpt_weights": "foundation-model-1x.weights",
    }
}

common_vpt_dir=os.path.join(config["common"]["path_data"], "VPT-models")

config_common_envs = {
    "num_train_epochs": 4,
    "batch_size": 1,
    "downsampling": 2,
    "lr": 1e-3,
    "l2_weight": 0,
    "entropy_weight": 0,
    "embedding_dim": 1024,
    "n_heads": 8,
    "n_layers": 6,
    "hidden_size": 128,
    "subset_training_len": 16,
    "save_every_n_epochs": 10,
    "skip_noops": True,
    "max_files_to_load": None,
    "button_encoder_num_actions": 125,
    "rollout_device": cuda_device, # "cpu", # alternative: cuda_device variable
    "vpt_model": os.path.join(common_vpt_dir,  config["common"]["vpt_model"]),
    "vpt_weights": os.path.join(common_vpt_dir, config["common"]["vpt_weights"]),
    "end_cut_episode_length": None,
    "end_episode_margin": 0,

}

common_embeddings_dir = os.path.join(config["common"]["path_data"], "embeddings", "foundation-model-1x.weights")
common_models_path = os.path.join(config["common"]["path_data"], "dt_models")
common_rollout_output_path = os.path.join(config["common"]["path_data"], "dt_rollouts")

config["envs"] = {
    "MineRLBasaltFindCave-v0": {**config_common_envs, **{
        "models_dir": os.path.join(common_models_path, "MineRLBasaltFindCave-v0"),
        "rollout_output_dir": os.path.join(common_rollout_output_path, "MineRLBasaltFindCave-v0"),
        "embeddings_dir": os.path.join(common_embeddings_dir, "MineRLBasaltFindCave-v0"),
        "button_act_csv_path": os.path.join( os.path.dirname(__file__), "resources", "MineRLBasaltFindCave-v0.frequencies.csv"),
        "batch_size":1, 
        "downsampling": 1,
        "num_train_epochs": 5,
        "n_layers": 6,
        "sequence_length": 64,
        "hidden_size": 256,
        "max_ep_len": 5000,
        "minibatch_samples": 4,
        "subset_training_len": 64, 
        "gamma":1.0,
        "scale_rewards":1,
        "mode": "delayed",
        "temperature_buttons": 1,
        "temperature_camera": 1,
        "temperature_esc": 1,
        "end_cut_episode_length": 64,
        "end_episode_margin": 5,
        "environment_seeds": ENV_TO_BASALT_2022_SEEDS["MineRLBasaltFindCave-v0"], 
        "rollout_max_steeps_per_seed": 3600,
        "env_targets": [100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100],
    }},
    "MineRLBasaltMakeWaterfall-v0": {**config_common_envs, **{
        "models_dir": os.path.join(common_models_path, "MineRLBasaltMakeWaterfall-v0"),
        "rollout_output_dir": os.path.join(common_rollout_output_path, "MineRLBasaltMakeWaterfall-v0"),
        "embeddings_dir": os.path.join(common_embeddings_dir, "MineRLBasaltMakeWaterfall-v0"),
        "button_act_csv_path": os.path.join( os.path.dirname(__file__), "resources", "MineRLBasaltMakeWaterfall-v0.act_frequencies.csv"),
        "downsampling": 1,
        "num_train_epochs": 5,
        "n_layers": 6,
        "sequence_length": 64,
        "hidden_size": 256,
        "max_ep_len": 5000,
        "minibatch_samples": 4, 
        "gamma":1.0,
        "scale_rewards":1,
        "mode": "delayed",
        "temperature_buttons": 1,
        "temperature_camera": 1,
        "temperature_esc": 1,
        "environment_seeds": ENV_TO_BASALT_2022_SEEDS["MineRLBasaltMakeWaterfall-v0"],
        "rollout_max_steeps_per_seed": 3000,
        "env_targets": [100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100],
    }},
    "MineRLBasaltBuildVillageHouse-v0": {**config_common_envs, **{
        "models_dir": os.path.join(common_models_path, "MineRLBasaltBuildVillageHouse-v0"),
        "rollout_output_dir": os.path.join(common_rollout_output_path, "MineRLBasaltBuildVillageHouse-v0"),
        "embeddings_dir": os.path.join(common_embeddings_dir, "MineRLBasaltBuildVillageHouse-v0"),
        "button_act_csv_path": os.path.join( os.path.dirname(__file__), "resources", "MineRLBasaltBuildVillageHouse-v0.act_frequencies.csv"),
        "max_files_to_load": None,
        "batch_size":8, 
        "downsampling": 1,
        "num_train_epochs": 100,
        "n_layers": 6,
        "sequence_length": 32,
        "hidden_size": 512,
        "max_ep_len": 5000,
        "minibatch_samples": 16, 
        "gamma":1.0,
        "scale_rewards":1,
        "mode": "delayed",
        "temperature_buttons": 1,
        "temperature_camera": 1,
        "temperature_esc": 1,
        "environment_seeds": ENV_TO_BASALT_2022_SEEDS["MineRLBasaltBuildVillageHouse-v0"], 
        "rollout_max_steeps_per_seed": 3600,
        "env_targets": [100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100],
    }},
}
