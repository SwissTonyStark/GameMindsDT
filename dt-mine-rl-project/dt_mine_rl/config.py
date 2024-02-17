import os
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
    "num_train_epochs": 10,
    "batch_size": 1,
    "downsampling": 2,
    "lr": 1e-3,
    "l2_weight": 0,
    "entropy_weight": 0,
    "embedding_dim": 1024,
    "n_heads": 4,
    "n_layers": 6,
    "hidden_size": 512,
    "subset_training_max_len": 16,
    "save_every_n_epochs": 10,
    "skip_noops": True,
    "max_files_to_load": None,
    "button_act_csv_path": os.path.join( os.path.dirname(__file__), "resources", "button_act_frequencies.csv"),
    "button_encoder_num_actions": 256,
    "rollout_device": cuda_device, # "cpu", # alternative: cuda_device variable
    "vpt_model": os.path.join(common_vpt_dir,  config["common"]["vpt_model"]),
    "vpt_weights": os.path.join(common_vpt_dir, config["common"]["vpt_weights"])

}

common_embeddings_dir = os.path.join(config["common"]["path_data"], "embeddings", "foundation-model-1x.weights")
common_models_path = os.path.join(config["common"]["path_data"], "dt_models")
common_rollout_output_path = os.path.join(config["common"]["path_data"], "dt_rollouts")

sequence_length_find_cave = 128
sequence_length_waterfall = 1024

config["envs"] = {
    "MineRLBasaltFindCave-v0": dict(**config_common_envs, {
        "models_dir": os.path.join(common_models_path, "MineRLBasaltFindCave-v0"),
        "rollout_output_dir": os.path.join(common_rollout_output_path, "MineRLBasaltFindCave-v0"),
        "embeddings_dir": os.path.join(common_embeddings_dir, "MineRLBasaltFindCave-v0"),
        "sequence_length": 128,
        "hidden_size": 2048,
        "max_ep_len": 5000,
        "minibatch_samples": 1, 
        "gamma":0.99,
        "scale_rewards":1,
        "mode": "delayed",
        "temperature_buttons": 1,
        "temperature_camera": 2,
        "temperature_esc": 1,
        "environment_seeds": None,
        "rollout_max_steeps_per_seed": 3000
    }),
    "MineRLBasaltMakeWaterfall-v0": dict(**config_common_envs, {
        "models_dir": os.path.join(common_models_path, "MineRLBasaltMakeWaterfall-v0"),
        "rollout_output_dir": os.path.join(common_rollout_output_path, "MineRLBasaltMakeWaterfall-v0"),
        "embeddings_dir": os.path.join(common_embeddings_dir, "MineRLBasaltMakeWaterfall-v0"),
        "sequence_length": 1024,
        "max_ep_len": 5000,
        "minibatch_samples": 2, 
        "gamma":0.99,
        "scale_rewards":1,
        "mode": "delayed",
        "temperature_buttons": 1,
        "temperature_camera": 2,
        "temperature_esc": 1,
        "environment_seeds": None,
        "rollout_max_steeps_per_seed": 3000
    })
}
