"""
Run multiple experiments on a single machine.
"""
import subprocess
from typing import List

import numpy as np

ALGOS = ["dqn"]
# ENVS = ["Acrobot-v1"]
ENVS = ["MountainCar-v0"]
N_SEEDS = 5
SEED_START = 100
EVAL_FREQ = 20000
N_EVAL_EPISODES = 5
WANDB_PROJECT_NAME = "seq-task-rl"
WANDB_GROUP = "acrobot_hypers"

for algo in ALGOS:
    for env_id in ENVS:
        log_folder = f"logs_{WANDB_GROUP}"
        for N in range(N_SEEDS):
            seed = SEED_START + N
            args = [
                "--algo",
                algo,
                "--env",
                env_id,
                "--seed",
                seed,
                "--eval-episodes",
                N_EVAL_EPISODES,
                "--eval-freq",
                EVAL_FREQ,
                "-f",
                log_folder,
                "--track",
                "--wandb-project-name",
                WANDB_PROJECT_NAME,
                "--wandb-group",
                WANDB_GROUP,
            ]
            arg_str_list: List[str] = list(map(str, args))

            ok = subprocess.call(["python", "train.py"] + arg_str_list)
