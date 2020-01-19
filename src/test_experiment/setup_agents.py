import os
import json

import sys
sys.path.append("../")
from utils import dict_product, iwt

with open("../MuJoCo.json") as f:
    BASE_CONFIG = json.load(f)

PARAMS = {
    # "game": ["Humanoid-v2"],
    "game": ["Walker2d-v2"],
    "mode": ["ppo"],
    "clip_eps": [1e8],
    "kl_penalty_coeff": [1],
    "kl_penalty_direction": ["old_to_new"],
    "out_dir": ["test_experiment/agents"],
    "norm_rewards": ["none"],
    # "initialization": ["xavier", "orthogonal"],
    "initialization": ["orthogonal"],
    "anneal_lr": [False],
    "value_clipping": [False],
    "clip_advantages": [3],
    # "ppo_lr_adam": iwt(1e-5, 2.9e-4, 7e-5, 5),
    "ppo_lr_adam": [3e-4] * 1,
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
