import gymnasium as gym
#import d4rl

# Create the environment
env = gym.make("BipedalWalker-v3")

# Automatically download and return the dataset
dataset = env.unwrapped.get_dataset()
print(dataset['observations']) # An (N, dim_observation)-dimensional numpy array of observations
print(dataset['actions']) # An (N, dim_action)-dimensional numpy array of actions
print(dataset['rewards']) # An (N,)-dimensional numpy array of rewards