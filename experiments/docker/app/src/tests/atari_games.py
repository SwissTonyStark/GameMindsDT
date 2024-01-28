import d3rlpy
import torch
import gym
import numpy as np
import os
from tqdm import tqdm

from config import config

from gym.wrappers import RecordVideo,AtariPreprocessing


def get_dt_default_config():
    return {"batch_size": 64, "learning_rate": 6e-4, "feature_size": 128, "num_heads": 8, "num_layers": 6, "temperature": 0.75}

def get_bc_default_config():
    return {"batch_size": 64}

def get_cql_default_config():
    return {"batch_size": 64}


def get_game_definitions():
    
    definitions = {
        "breakout": {"env_name": "BreakoutNoFrameskip-v4", "dataset_env_name": "breakout", "num_stack": 3, "target_return": 250, "context_size": 50, "max_timestep": 3500},
        "qbert": {"env_name": "QbertNoFrameskip-v4", "dataset_env_name": "qbert", "num_stack": 3, "target_return": 20000, "context_size": 30, "max_timestep": 4000},
        "pong": {"env_name": "PongNoFrameskip-v4", "dataset_env_name": "pong", "num_stack": 3, "target_return": 20, "context_size": 50, "max_timestep": 4000},
        "seaquest": {"env_name": "SeaquestNoFrameskip-v4", "dataset_env_name": "seaquest", "num_stack": 3, "target_return": 5000, "context_size": 30, "max_timestep": 3500}
    }

    return definitions

def get_enabled_games():
    return ["breakout", "qbert", "pong", "seaquest"], get_game_definitions()


def get_dt_atari_model(dt_config, context_size = 30, max_timestep=100000):
    dt = d3rlpy.algos.DiscreteDecisionTransformerConfig(
        batch_size=dt_config["batch_size"],
        context_size=context_size,
        learning_rate=dt_config["learning_rate"],
        activation_type="gelu",
        embed_activation_type="tanh",
        encoder_factory=d3rlpy.models.PixelEncoderFactory(
            feature_size=dt_config["feature_size"], exclude_last_activation=True
        ),  # Nature DQN
        num_heads=dt_config["num_heads"],
        num_layers=dt_config["num_layers"],
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

    return dt

def train_dt_atari_game(game_definition, dt_config):

    dataset_env_name = game_definition["dataset_env_name"]
    context_size = game_definition["context_size"]
    target_return = game_definition["target_return"]
    num_stack = game_definition["num_stack"]
    max_timestep = game_definition["max_timestep"]

    if "training_steps" in game_definition:
        training_steps = game_definition["training_steps"]
    else:
        training_steps = None


    # If exists, load model
    model_path = os.path.join(config["models_path"], f"atari-{dataset_env_name}-dt.d3")
    if os.path.exists(model_path):
        print(f"Model already exists for {dataset_env_name}. Skipped training.")
        return
 
    dataset, env = d3rlpy.datasets.get_atari(
        f"{dataset_env_name}-expert-v0",
        num_stack=num_stack
    )

    max_time_step_dataset = 0
    for episode in dataset.episodes:
        max_time_step_dataset = max(max_time_step_dataset, episode.transition_count + 1)
    print("max_time_step_dataset", max_time_step_dataset)

    dt = get_dt_atari_model(dt_config, context_size=context_size, max_timestep=max_timestep)

    if training_steps:
        n_steps_per_epoch = training_steps // 5
    else:
        n_steps_per_epoch = dataset.transition_count // dt_config["batch_size"] // 5
    n_steps = n_steps_per_epoch * 5

    seed = config["seed"]

    dt.fit(
        dataset,
        n_steps=n_steps,
        n_steps_per_epoch=n_steps_per_epoch,
        eval_env=env,
        eval_target_return=target_return,
        eval_action_sampler=d3rlpy.algos.SoftmaxTransformerActionSampler(
            temperature=dt_config["temperature"],
        ),
        experiment_name=f"DiscreteDT_atari_{dataset_env_name}_{ seed }"
    )

    dt.save_model(os.path.join(config["models_path"], f"atari-{dataset_env_name}-dt.d3"))

    return dt

def load_model(dataset_env_name, algo, env, algo_key):
    model_path = os.path.join(config["models_path"], f"atari-{dataset_env_name}-{algo_key}.d3")
    if os.path.exists(model_path):
        algo.build_with_env(env)
        algo.load_model(model_path)

    return algo

def create_env(env_key, num_stack=1, max_episode_steps=None, is_video=False):
    
    render_mode = None

    if is_video == True:

        render_mode = 'rgb_array'

    env = gym.make(env_key, max_episode_steps=max_episode_steps, render_mode=render_mode)
    env = AtariPreprocessing(env,  terminal_on_life_loss=False)
    env = d3rlpy.datasets.FrameStack(env, num_stack=num_stack)

    return env

def generate_video_atari_dt(dt, game_definition, dt_config):

    env_name = game_definition["env_name"]
    dataset_env_name = game_definition["dataset_env_name"]
    target_return = game_definition["target_return"]
    num_stack = game_definition["num_stack"]
    
    env = create_env(env_name, num_stack=num_stack, is_video=True)

    env = RecordVideo(env, os.path.join(config["videos_path"], f"video-atari-{env_name}-dt"))

    dt = load_model(dataset_env_name, dt, env, "dt")

    # wrap as stateful actor for interaction
    actor = dt.as_stateful_wrapper(
        target_return=target_return,
        action_sampler=d3rlpy.algos.SoftmaxTransformerActionSampler(
        temperature=dt_config["temperature"],
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

def train_dt_atari_games(enabled_games, game_definitions):

    dt_config = get_dt_default_config()
    
    for game in enabled_games:
        train_dt_atari_game(game_definitions[game], dt_config)

def train_algo_atari_game(algo, algo_key, game_definition, cql_config):

    dataset_env_name = game_definition["dataset_env_name"]
    seed = config["seed"]
    num_stack = game_definition["num_stack"]

    if "training_steps" in game_definition:
        training_steps = game_definition["training_steps"]
    else:
        training_steps = None


    # If exists, load model
    model_path = os.path.join(config["models_path"], f"atari-{dataset_env_name}-{algo_key}.d3")
    if os.path.exists(model_path):
        print(f"Model already exists for {dataset_env_name} - model {algo_key}. Skipped training.")
        return
 
    dataset, env = d3rlpy.datasets.get_atari(
        f"{dataset_env_name}-expert-v0",
        num_stack=num_stack
    )

    if training_steps:
        n_steps_per_epoch = training_steps // 5
    else:
        n_steps_per_epoch = dataset.transition_count // cql_config["batch_size"] // 5
    n_steps = n_steps_per_epoch * 5

    algo.fit(
        dataset,
        n_steps=n_steps,
        n_steps_per_epoch=n_steps_per_epoch,
        experiment_name=f"Discrete_atari_{dataset_env_name}-{algo_key}_{ seed }")

    algo.save_model(os.path.join(config["models_path"], f"atari-{dataset_env_name}-{algo_key}.d3"))



def train_bc_atari_games(enabled_games, game_definitions):
    
    bc_config = get_bc_default_config()

    for game in enabled_games:
        bc = d3rlpy.algos.DiscreteBCConfig(batch_size=bc_config["batch_size"],).create(device=config["device"])
        train_algo_atari_game(bc, "bc", game_definitions[game], bc_config)

def train_cql_atari_games(enabled_games, game_definitions):
    
    cql_config = get_cql_default_config()
    
    for game in enabled_games:
        cql = d3rlpy.algos.DiscreteCQLConfig(batch_size=cql_config["batch_size"],).create(device=config["device"])
        train_algo_atari_game(cql, "cql", game_definitions[game], cql_config)


def generate_atari_videos(enabled_games, game_definitions):

    dt_config = get_dt_default_config()

    for game in enabled_games:
        dt = get_dt_atari_model(dt_config, context_size=game_definitions[game]["context_size"], max_timestep=game_definitions[game]["max_timestep"])
        generate_video_atari_dt(dt, game_definitions[game], dt_config)

def evaluate_policy(env, actor, explorer=None, n_trials=10):
    success = 0
    sum_reward = 0
    for _ in tqdm(range(n_trials)):
        
        if (explorer is None):
            actor.reset()
        obs, reward = env.reset()[0], 0.0

        done, truncated = False, False
        while not (done or truncated):
            if explorer is not None:
                x = np.expand_dims(obs, axis=0)
                action = explorer.sample(actor, x, 0)[0]
            else:
                action = actor.predict(obs, reward)
            obs, reward, done, truncated, _ = env.step(action)
            if done and reward > 0:
                success += 1
            sum_reward += reward
    return sum_reward / n_trials, success / n_trials


def evaluate_atari_games(enabled_games, game_definitions, trials=5, epsilon=0.1):
    
    dt_config = get_dt_default_config()
    temperature = dt_config["temperature"]

    results_path = os.path.join(config["results_path"], "atari_results.npy")

    if os.path.exists(results_path):
        return np.load(results_path, allow_pickle=True)

    results = []

    for game in enabled_games:

        env_name = game_definitions[game]["env_name"]
        num_stack = game_definitions[game]["num_stack"]
        dataset_env_name = game_definitions[game]["dataset_env_name"]
        context_size = game_definitions[game]["context_size"]
        max_timestep = game_definitions[game]["max_timestep"]
        target_return = game_definitions[game]["target_return"]

        env = create_env(env_name, num_stack=num_stack)

        bc_config = get_bc_default_config()
        bc = d3rlpy.algos.DiscreteBCConfig().create(device=config["device"])
        bc = load_model(dataset_env_name, bc, env, "bc")
        explorer = d3rlpy.algos.ConstantEpsilonGreedy(epsilon=epsilon)
        bc_reward_score, _ = evaluate_policy(env, bc, explorer, n_trials=trials)

        cql_config = get_cql_default_config()
        cql = d3rlpy.algos.DiscreteCQLConfig().create(device=config["device"])
        cql = load_model(dataset_env_name, cql, env, "cql")
        explorer = d3rlpy.algos.ConstantEpsilonGreedy(epsilon=epsilon)
        cql_reward_score, _ = evaluate_policy(env, cql, explorer, n_trials=trials)

        dt = get_dt_atari_model(dt_config, context_size=context_size, max_timestep=max_timestep)
        dt = load_model(dataset_env_name, dt, env, "dt")
        actor_dt = dt.as_stateful_wrapper(
            target_return=target_return,
            action_sampler=d3rlpy.algos.SoftmaxTransformerActionSampler(temperature=temperature,)
        )
        dt_reward_score, _ = evaluate_policy(env, actor_dt, n_trials=trials)

        print(f"Game: {env_name}")
        print(f"DT: {dt_reward_score} - BC: {bc_reward_score} - CQL: {cql_reward_score}")

        results.append({"game": game, "dt": dt_reward_score, "bc": bc_reward_score, "cql": cql_reward_score})

    np.save(results_path, results)

    return results

# Modified function to create separate bar charts for each game
def create_game_chart(results, game_index, ax):

    game_results = results[game_index]
    algorithms = ["dt", "bc", "cql"]
    
    scores = [game_results[alg] for alg in algorithms]
    x = np.arange(len(algorithms))

    ax.bar(x, scores, width=0.5)

    ax.set_ylabel('Scores')
    ax.set_title(f'Scores for {game_results["game"]}')
    ax.set_xticks(x)
    ax.set_xticklabels(algorithms)
    ax.set_ylim(0, max(scores) + max(scores) * 0.05)



def graph_results(results):

    import matplotlib.pyplot as plt

    num_games = len(results)
    fig, axs = plt.subplots(1, num_games, figsize=(5 * num_games, 5))

    for i in range(num_games):
        create_game_chart(results, i, axs[i])

    fig.tight_layout()

    fig.savefig(os.path.join(config["results_path"], "atari_results.png"))

    plt.show()


def run_atari_tests():

    d3rlpy.seed(config["seed"])
    
    enabled_games, game_definitions = get_enabled_games()

    train_dt_atari_games(enabled_games, game_definitions)
    train_bc_atari_games(enabled_games, game_definitions)
    train_cql_atari_games(enabled_games, game_definitions)
    results = evaluate_atari_games(enabled_games, game_definitions, trials=100)
    graph_results(results)
    generate_atari_videos(enabled_games, game_definitions)
    
if __name__ == '__main__':
    run_atari_tests()