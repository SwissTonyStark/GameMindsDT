from minerl.herobraine.env_specs.basalt_specs import BasaltBaseEnvSpec, FindCaveEnvSpec, MakeWaterfallEnvSpec, PenAnimalsVillageEnvSpec, VillageMakeHouseEnvSpec
from minerl.herobraine.hero import handlers
from minerl.herobraine.hero.mc import ALL_ITEMS

from vpt_lib.agent import AGENT_NUM_BUTTON_ACTIONS, AGENT_NUM_CAMERA_ACTIONS

from gym import spaces

import numpy as np
import json
import pickle

KEYS_OF_INTEREST = ['equipped_items', 'life_stats', 'location_stats', 'use_item', 'drop', 'pickup', 'break_item', 'craft_item', 'mine_block', 'damage_dealt', 'entity_killed_by', 'kill_entity', 'full_stats']

KEYS_FOR_TRANSITIONS = ["obs", "next_obs", "acts", "rewards", "dones", "infos"]

# Maximum number of transitions to load. Using a fixed array size avoids expensive recreation of arrays.
# This is hardcoded for `downsampling`=2.
# This is suitable for ~5GB of RAM.
MAX_DATA_SIZE = 1_00_000

IMAGE_RESOLUTION = (640, 360)

AGENT_DT_NUM_BUTTON_ACTIONS = AGENT_NUM_BUTTON_ACTIONS
AGENT_DT_NUM_CAMERA_ACTIONS = AGENT_NUM_CAMERA_ACTIONS
AGENT_DT_NUN_ESC_BUTTON = 2
 
# Manual mapping so that below patching works
ENV_NAME_TO_SPEC = {
    "MineRLBasaltFindCave-v0": FindCaveEnvSpec,
    "MineRLBasaltMakeWaterfall-v0": MakeWaterfallEnvSpec,
    "MineRLBasaltCreateVillageAnimalPen-v0": PenAnimalsVillageEnvSpec,
    "MineRLBasaltBuildVillageHouse-v0": VillageMakeHouseEnvSpec,
}

# These are the seeds per environment models were evaluated on
ENV_TO_BASALT_2022_SEEDS = {
    "MineRLBasaltFindCave-v0": [14169, 65101, 78472, 76379, 39802, 95099, 63686, 49077, 77533, 31703, 73365],
    "MineRLBasaltMakeWaterfall-v0": [95674, 39036, 70373, 84685, 91255, 56595, 53737, 12095, 86455, 19570, 40250],
    "MineRLBasaltCreateVillageAnimalPen-v0": [21212, 85236, 14975, 57764, 56029, 65215, 83805, 35884, 27406, 5681265, 20848],
    "MineRLBasaltBuildVillageHouse-v0": [52216, 29342, 67640, 73169, 86898, 70333, 12658, 99066, 92974, 32150, 78702],
}

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

def build_obs_and_act_gym_spaces(dataset, n_reduced_button_actions=None):
    observation_space = spaces.Box(low=-float("inf"), high=float("inf"), shape=(dataset[0]["obs"].shape[1],))
    # 2 is for ESC button

    if n_reduced_button_actions is not None:
        action_space = spaces.MultiDiscrete([n_reduced_button_actions, AGENT_DT_NUM_CAMERA_ACTIONS, 2])
    else: 
        action_space = spaces.MultiDiscrete([AGENT_DT_NUM_BUTTON_ACTIONS, AGENT_DT_NUM_CAMERA_ACTIONS, 2])

    return observation_space, action_space


def load_model_parameters(path_to_model_file):
    agent_parameters = pickle.load(open(path_to_model_file, "rb"))
    policy_kwargs = agent_parameters["model"]["args"]["net"]["args"]
    pi_head_kwargs = agent_parameters["model"]["args"]["pi_head_opts"]
    pi_head_kwargs["temperature"] = float(pi_head_kwargs["temperature"])
    return policy_kwargs, pi_head_kwargs
