import os
import json

import sys
sys.path.append("../")
from utils import dict_product, iwt

with open("../MuJoCo.json") as f:
    BASE_CONFIG = json.load(f)

PARAMS = {
    "policy_net_type": ["CtsPolicy"],
    "game": ["Walker2d-v2", "Humanoid-v2", "Swimmer-v2", "Hopper-v2",
        "HalfCheetah-v2", "InvertedPendulum-v2", "Reacher-v2",
        "InvertedDoublePendulum-v2"],
    "mode": ["ppo"],
    "clip_eps": [0.2],
    "kl_penalty_coeff": [0.0],
    "ppo_lr_adam": [3e-4] * 1,
    "kl_penalty_direction": ["new_to_old"],
    "clip_action": [True],
    "strict_action_bounds": [False],
    "adjust_init_std": [False],
    "out_dir": ["action_logging/agents"],
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
