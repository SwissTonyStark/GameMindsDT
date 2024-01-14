import sys
import os
import d3rlpy

from config import config
from src.tests.pendulum import run_pendulum
from src.tests.atari_breakout import run_atari_breakout
import logging

seed=1
d3rlpy.seed(config["seed"])

# Directory creation
isExist = os.path.exists(config["models_path"])
if not isExist:
    os.makedirs(config["models_path"])

isExist = os.path.exists(config["videos_path"])
if not isExist:
    os.makedirs(config["videos_path"])

if sys.argv[1] == "test_pendulum":
    run_pendulum()
elif sys.argv[1] == "test_atari_breakout":
    run_atari_breakout()
else:
    logging.error("Invalid argument")
    exit(1)

