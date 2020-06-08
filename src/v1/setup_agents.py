import os
import json

import sys
sys.path.append("../")
from utils import dict_product, iwt

with open("../MuJoCo.json") as f:
    BASE_CONFIG = json.load(f)

PARAMS = {
    "policy_net_type": ["CtsBetaPolicy", "CtsPolicy"],
    "game": ["Humanoid-v1", "MountainCarContinuous-v0", "Pendulum-v0",
        "InvertedPendulum-v1", "InvertedDoublePendulum-v1"],
    "mode": ["ppo"],
    "clip_eps": [0.2, 1e8],
    "kl_penalty_coeff": [0., 3.],
    "ppo_lr_adam": [3e-4] * 2,
    "kl_penalty_direction": ["new_to_old"],
    "out_dir": ["v1/agents"],
    "norm_rewards": ["returns"],
    "advanced_logging": [False],
}

all_configs = [{**BASE_CONFIG, **p} for p in dict_product(PARAMS)]
if os.path.isdir("agent_configs/") or os.path.isdir("agents/"):
    raise ValueError("Please delete the 'agent_configs/' and 'agents/' directories")
os.makedirs("agent_configs/")
os.makedirs("agents/")

for i, config in enumerate(all_configs):
    with open(f"agent_configs/{i}.json", "w") as f:
        json.dump(config, f)
