import argparse
import os
import pprint

from dt_models.dt_model_gm import TrainableDTGM

from dt_models.dt_model_common import ActEncoderDecoder

from lib.dataset import EpisodeDataset

from transformers import TrainingArguments, Trainer

from torch.utils.data import random_split

from dt_models.dt_model_common import DecisionTransformerGymEpisodeCollator

from config import config


def main(args):

    env_key = args.env

    app_config = config["envs"][env_key]
    
    pprint.PrettyPrinter(indent=4,sort_dicts=True).pprint(app_config)

    # If training dir already exists and flag is set, do not train
    models_dir = app_config["models_dir"]
    if os.path.exists(models_dir) and config["common"]["skip_if_exists"]:
        print(f"Output directory {models_dir} already exists. Skipping training.")
        return

    act_button_encoder = None

    if (app_config["button_act_csv_path"] is not None):
        act_button_encoder = ActEncoderDecoder(app_config["button_act_csv_path"], app_config["button_encoder_num_actions"], app_config["embeddings_dir"])
    
    episode_dataset = EpisodeDataset(
        app_config["embeddings_dir"],
        app_config["embedding_dim"],
        max_files_to_load=app_config["max_files_to_load"],
        downsampling=app_config["downsampling"],
        skip_noops=app_config["skip_noops"],
        act_button_encoder=act_button_encoder,
        end_cut_episode_length=app_config["end_cut_episode_length"],
        end_episode_margin=app_config["end_episode_margin"])
    
    state_dim, action_dim = episode_dataset.get_state_and_act_dim()

    dataset_size = len(episode_dataset)
    train_size = int(0.9 * dataset_size)
    test_size = dataset_size - train_size
    
    train_dataset, eval_dataset = random_split(episode_dataset, [train_size, test_size])

    collator = DecisionTransformerGymEpisodeCollator( 
        state_dim, action_dim, app_config["subset_training_len"], app_config["max_ep_len"], app_config["minibatch_samples"], app_config["gamma"], app_config["scale_rewards"])

    AgentClass = app_config["agent_implementation"] 
    model = AgentClass.from_config(**app_config)

    os.makedirs(models_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=models_dir,
        remove_unused_columns=False,
        num_train_epochs=app_config["num_train_epochs"],
        per_device_train_batch_size=app_config["batch_size"],
        per_device_eval_batch_size=app_config["batch_size"],
        learning_rate=app_config["lr"],
        optim="adamw_torch",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
    )

    trainer.train()

    # Save the final model
    trainer.save_model(models_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train a DT agent on VPT embeddings for the BASALT Benchmark")
    parser.add_argument("--env", type=str, required=True, help="Environment to train")
    args = parser.parse_args()
    main(args)