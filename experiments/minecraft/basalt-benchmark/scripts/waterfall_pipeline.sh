#!/bin/bash
# Run through whole setup, training and rollout pipeline with quick settings to test that everything works
# This does following steps:
#  - Download max 4GB of demonstration data (+ 1GB for the VPT models)
#  - Embed the data
#  - Train models with only few epochs
#  - Rollout models on MineRLBasalt* environments for only few seconds to produce short videos
# If everything is succesful, you should see the videos in the `pipeline_test_data/rollouts` directory.
set -e

# Check that we are in the root of the repository
if [ ! -d "scripts" ]; then
    echo "This script should be run from the root of the repository with 'bash ./scripts/test_pipeline.sh'"
    exit 1
fi

# Download limited amount of demonstration data
bash ./scripts/download_demonstration_waterfall_data.sh /mnt/hdd_01/PROJECTS/fib_postgraduate/minerl/basalt-benchmark/pipeline_test_data

# Embed the data
bash ./scripts/embed_trajectories_waterfall.sh /mnt/hdd_01/PROJECTS/fib_postgraduate/minerl/basalt-benchmark/pipeline_test_data

# Train models with only five epochs
bash ./scripts/train_bc.sh /mnt/hdd_01/PROJECTS/fib_postgraduate/minerl/basalt-benchmark/pipeline_test_data 1000

# Rollout models on MineRLBasalt* environments for only 5 seconds to produce short videos
# Each second is 20 steps
bash ./scripts/rollout_bc_waterfall.sh /mnt/hdd_01/PROJECTS/fib_postgraduate/minerl/basalt-benchmark/pipeline_test_data 2000