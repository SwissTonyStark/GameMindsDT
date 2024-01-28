
import os
import torch

base_path = os.path.dirname(os.path.abspath(__file__))
device = "cuda:0" if torch.cuda.is_available() else "cpu"

config = {
    "base_path": base_path,
    "models_path": os.path.join(base_path,"models"),
    "videos_path": os.path.join(base_path,"videos"),
    "results_path": os.path.join(base_path,"results"),
    "seed": 1,
    "device": device,
}

