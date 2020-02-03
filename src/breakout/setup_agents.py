import os
import json

import sys
sys.path.append("../")
from utils import dict_product, iwt

with open("../Atari.json") as f:
    BASE_CONFIG = json.load(f)

PARAMS = {
    "game": ["Breakout-v0"],
    "clip_eps": [0.2, 1e8],
    "kl_penalty_coeff": [0, 3],
    "ppo_lr_adam": [2.5e-4] * 4,
    "kl_penalty_direction": ["old_to_new", "new_to_old"],
    "out_dir": ["breakout/agents"],
    "norm_rewards": ["returns"],
    "initialization": ["orthogonal"],
    "anneal_lr": [True],
    "val_lr": [2e-5],
    "cpu": [True],
    "advanced_logging": [True],
}

all_configs = [{**BASE_CONFIG, **p} for p in dict_product(PARAMS)]
if os.path.isdir("agent_configs/") or os.path.isdir("agents/"):
    raise ValueError("Please delete the 'agent_configs/' and 'agents/' directories")
os.makedirs("agent_configs/")
os.makedirs("agents/")

for i, config in enumerate(all_configs):
    with open(f"agent_configs/{i}.json", "w") as f:
        json.dump(config, f)
