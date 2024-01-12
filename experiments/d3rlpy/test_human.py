import gymnasium as gym
from gym.utils.play import play
import numpy as np


play(gym.make("MiniGrid-DoorKey-5x5-v0", render_mode="rgb_array"), keys_to_action={  
                                               "w": np.array([2]),
                                               "a": np.array([0]),
                                               "d": np.array([1]),
                                               "s": np.array([3]),
                                               "p": np.array([5]),
                                              }, noop=np.array([6]))