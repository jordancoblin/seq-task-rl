import os

import numpy as np

from rl_zoo3.algorithms.a2c import A2C
from rl_zoo3.algorithms.common.utils import get_system_info
from rl_zoo3.algorithms.ddpg import DDPG
from rl_zoo3.algorithms.dqn import DQN
from rl_zoo3.algorithms.her.her_replay_buffer import HerReplayBuffer
from rl_zoo3.algorithms.ppo import PPO
from rl_zoo3.algorithms.sac import SAC
from rl_zoo3.algorithms.td3 import TD3

# Small monkey patch so gym 0.21 is compatible with numpy >= 1.24
# TODO: remove when upgrading to gym 0.26
np.bool = bool  # type: ignore[attr-defined]

# Read version from file
version_file = os.path.join(os.path.dirname(__file__), "version.txt")
with open(version_file) as file_handler:
    __version__ = file_handler.read().strip()


def HER(*args, **kwargs):
    raise ImportError(
        "Since Stable Baselines 2.1.0, `HER` is now a replay buffer class `HerReplayBuffer`.\n "
        "Please check the documentation for more information: https://stable-baselines3.readthedocs.io/"
    )


__all__ = [
    "A2C",
    "DDPG",
    "DQN",
    "PPO",
    "SAC",
    "TD3",
    "HerReplayBuffer",
    "get_system_info",
]
