import d3rlpy
import torch
import gym
import numpy as np
import os

from config import config

from gym.wrappers import RecordVideo,AtariPreprocessing


def train_atari_breakout(num_stack = 3, batch_size = 64, context_size = 30):
    dataset, env = d3rlpy.datasets.get_atari_transitions(
        "breakout",
        fraction=0.01,
        index=0,
        num_stack=num_stack,
        sticky_action=False
    )

    max_timestep = 0
    for episode in dataset.episodes:
        max_timestep = max(max_timestep, episode.transition_count + 1)

    dt = d3rlpy.algos.DiscreteDecisionTransformerConfig(
            batch_size=batch_size,
            context_size=context_size,
            learning_rate=6e-4,
            activation_type="gelu",
            embed_activation_type="tanh",
            encoder_factory=d3rlpy.models.PixelEncoderFactory(
                feature_size=128, exclude_last_activation=True
            ),  # Nature DQN
            num_heads=8,
            num_layers=6,
            attn_dropout=0.1,
            embed_dropout=0.1,
            optim_factory=d3rlpy.models.GPTAdamWFactory(
                betas=(0.9, 0.95),
                weight_decay=0.1,
            ),
            clip_grad_norm=1.0,
            warmup_tokens=512 * 20,
            final_tokens=2 * 500000 * context_size * 3,
            observation_scaler=d3rlpy.preprocessing.PixelObservationScaler(),
            max_timestep=max_timestep,
            position_encoding_type=d3rlpy.PositionEncodingType.GLOBAL,
        ).create(device=config["device"])

    target_return=1000

    # offline training
    n_steps_per_epoch = dataset.transition_count // batch_size
    n_steps = n_steps_per_epoch * 5

    seed = config["seed"]
    
    dt.fit(
        dataset,
        n_steps=n_steps,
        n_steps_per_epoch=n_steps_per_epoch,
        eval_env=env,
        eval_target_return=target_return,
        eval_action_sampler=d3rlpy.algos.SoftmaxTransformerActionSampler(
            temperature=1.0,
        ),
        experiment_name=f"DiscreteDT_atari_breakout_{ seed }"
    )

    dt.save_model(os.path.join(config["models_path"], "atari-breakout-dt.d3"))


    return dt

def generate_video_atari_breakout(dt, num_stack = 3):

    env = gym.make('BreakoutNoFrameskip-v4', render_mode='rgb_array', repeat_action_probability=0)
    env = AtariPreprocessing(env,  terminal_on_life_loss=False)

    env =  d3rlpy.datasets.FrameStack(env, num_stack=num_stack)

    env = RecordVideo(env, os.path.join(config["videos_path"], "video-breakout-dt"))

    # wrap as stateful actor for interaction
    actor = dt.as_stateful_wrapper(
        target_return=1000,
        action_sampler=d3rlpy.algos.SoftmaxTransformerActionSampler(
        temperature=1.0,
    ))

    n_trials: int = 1

    episode_rewards = []
    for _ in range(n_trials):
        actor.reset()
        observation, reward = env.reset()[0], 0.0
        episode_reward = 0.0

        frame = 0
        while True:
            action = actor.predict(observation, reward)
            observation, _reward, done, truncated, _ = env.step(action)
            reward = float(_reward)
            episode_reward += reward

            if done or truncated:
                break
            frame += 1
        episode_rewards.append(episode_reward)

def run_atari_breakout(num_stack = 3):
    dt = train_atari_breakout(num_stack)
    generate_video_atari_breakout(dt, num_stack)


if __name__ == '__main__':
  run_atari_breakout()