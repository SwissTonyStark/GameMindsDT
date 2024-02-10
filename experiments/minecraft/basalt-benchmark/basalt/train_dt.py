import argparse
import os
import random

from basalt.utils.embedded_dt_data_loading import EpisodeDataset, load_data_for_dt_from_path

from transformers import DecisionTransformerConfig, TrainingArguments, Trainer

from basalt.dt_model import DecisionTransformerGymEpisodeCollator, TrainableDT

from torch.utils.data import random_split

def add_experiment_specific_args(parser):
    parser.add_argument("--embeddings_dir", type=str, required=True, help="Path to the directory that contains the data embeddings")
    parser.add_argument("--output_dir", type=str, required=True, help="Where to store the experiment results")

    parser.add_argument("--max_files_to_load", type=int, default=None, help="Maximum number of embedding files to load. Takes the first ones.")
    parser.add_argument("--downsampling", type=int, default=1, help="Stride for loading a samples from a file (e.g. 2 -> take every other sample).")
    parser.add_argument("--skip_noops", action="store_true", help="If given, ignore actions that are no-ops.")
    parser.add_argument("--n_epochs", type=int, default=10000, help="number of epochs to train for")
    parser.add_argument("--batch_size", type=int, default=16, help="batch size for training")

    parser.add_argument("--learning_rate", type=float, default=1e-4, help="learning rate for training")
    parser.add_argument("--l2_weight", type=float, default=1e-5, help="L2 loss weight for training")
    parser.add_argument("--entropy_weight", type=float, default=1e-3, help="Entropy weight")

    parser.add_argument("--n_layers", type=int, default=2, help="Number of layers in the MLP")

    parser.add_argument("--embedding_dim", type=int, default=None, help="Embedding dimension for the data embeddings")

    parser.add_argument("--skip_if_exists", action="store_false", help="If set, will not train if the output directory already exists")

    parser.add_argument("--save_every_epochs", type=int, default=1, help="Save the model every n epochs")
    parser.add_argument("--log_every_batches", type=int, default=500, help="How often should we log metrics (in batches)")

    parser.add_argument("--seed", type=int, default=random.randint(0, 1000000), help="Random seed to use")

def main(args):
    # If training dir already exists and flag is set, do not train
    if os.path.exists(args.output_dir) and args.skip_if_exists:
        print(f"Output directory {args.output_dir} already exists. Skipping training.")
        return

    #dataset, observation_space, action_space = load_data_for_dt_from_path(
    #    args.embeddings_dir,
    #    args.embedding_dim,
    #    max_files_to_load=args.max_files_to_load,
    #    downsampling=args.downsampling,
    #    skip_noops=args.skip_noops
    #)

    #print("action_space", action_space)

    episode_dataset = EpisodeDataset(
        args.embeddings_dir,
        args.embedding_dim,
        max_files_to_load=args.max_files_to_load,
        downsampling=args.downsampling,
        skip_noops=args.skip_noops)
    
    state_dim, act_dim, observation_space, action_space = episode_dataset.get_state_and_act_dim()

    dataset_size = len(episode_dataset)
    train_size = int(0.9 * dataset_size)
    test_size = dataset_size - train_size
    
    train_dataset, eval_dataset = random_split(episode_dataset, [train_size, test_size])

    collator = DecisionTransformerGymEpisodeCollator(state_dim=state_dim, act_dim=act_dim)
    
    #collator = DecisionTransformerGymDataCollator(episode_dataset)

    # Initializing a DecisionTransformer configuration
    config = DecisionTransformerConfig(
        n_head=8,
        n_layer=6,
        hidden_size=256,
        n_positions=128*3,
        max_ep_len=10000,
        state_dim=collator.state_dim, 
        act_dim=collator.act_dim)
    model = TrainableDT(config)


    os.makedirs(args.output_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        remove_unused_columns=False,
        num_train_epochs=args.n_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=1e-4,
        warmup_ratio=0.1,
        optim="adamw_torch",
        max_grad_norm=0.25,
        evaluation_strategy="epoch"
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
    trainer.save_model(args.output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train a DT agent on VPT embeddings for the BASALT Benchmark")
    add_experiment_specific_args(parser)
    args = parser.parse_args()
    main(args)