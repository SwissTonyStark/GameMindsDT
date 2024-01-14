import gym
import numpy as np
from gym.wrappers import RecordVideo
import d3rlpy
import os

from config import config
      
def train_pendulum():

  # get Pendulum dataset
  dataset, env = d3rlpy.datasets.get_pendulum()

  # Setup Decision Transformer
  dt = d3rlpy.algos.DecisionTransformerConfig().create(device=config["device"])

  # offline training
  seed = config["seed"]
  dt.fit(
    dataset,
    n_steps=10000,
    n_steps_per_epoch=1000,
    eval_env=env,
    eval_target_return=0, 
    experiment_name=f"DiscreteDT_pendulum_{ seed }"
  )

  dt.save_model(os.path.join(config["models_path"], "pendulum-dt.d3"))

  return dt

def generate_video_pendulum(dt):
   # wrap RecordVideo wrapper
  env = RecordVideo(gym.make("Pendulum-v1", render_mode="rgb_array"), os.path.join(config["videos_path"], "video-pendulum-dt"))

  # wrap as stateful actor for interaction
  actor = dt.as_stateful_wrapper(target_return=0)

  # interaction
  observation, reward = env.reset(), 0.0
  observation = observation[0]
  while True:
      action = actor.predict(observation, reward)
      observation, reward, done, truncated, _ = env.step(action)
      if done or truncated:
          break

def run_pendulum():
  dt = train_pendulum()
  generate_video_pendulum(dt)

  
if __name__ == '__main__':
  run_pendulum()