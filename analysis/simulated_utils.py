import os
import sys

from gym.spaces.box import Box as Continuous
from gym.spaces.discrete import Discrete
import gym
import numpy as np
import pandas as pd
import torch

module_path = os.path.abspath(os.path.join('../src'))
if module_path not in sys.path:
    sys.path.append(module_path)
from policy_gradients.steps import ppo_step
from policy_gradients.models import DiscPolicy, CtsPolicy, CtsBetaPolicy
from policy_gradients.torch_utils import Parameters

new_to_old_params = {'kl_penalty_direction': 'new_to_old', 'kl_penalty_coeff': 3., 'clip_eps': None}
old_to_new_params = {'kl_penalty_direction': 'old_to_new', 'kl_penalty_coeff': 3., 'clip_eps': None}
clip_params = {'kl_penalty_coeff': 0., 'clip_eps': 0.2}
plain_params = {'kl_penalty_coeff': 0., 'clip_eps': None}

comparisons = {
    'kl_flip': new_to_old_params,
    'kl_orig': old_to_new_params,
    'clip': clip_params,
    'plain_pg': plain_params
}

base_params = {
    'ppo_epochs': 1,
    'num_minibatches': 16,
    'clip_advantages': None,
    'kl_penalty_direction': 'old_to_new',
    'entropy_coeff': 0.0,
    'share_weights': False,
    'clip_grad_norm': -1,
    'batch_size': 128,
    'max_episode_len': 1,
    'lr': 1e-2,
}


class SimpleEnv(gym.Env):
    '''
    A simple single-state environment.
    One-dimensional continuous action between [-10, 10], or discrete actions.
    Exponentially decaying reward function centered at action = 5.
    '''
    def __init__(self, action_dim=1, obs_dim=1, noise_std=0.1, is_discrete=False):
        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self.noise_std = noise_std
        if is_discrete:
            self.action_space = Discrete(action_dim)
        else:
            low = -10. * np.ones(action_dim)
            high = -low
            self.action_space = Continuous(low=low, high=high, dtype=np.float32) 
        self.observation_space = Continuous(low=np.zeros(obs_dim),
                high=np.ones(obs_dim), dtype=np.float32) 
        self.reset()

    def reset(self):
        obs = np.random.rand(self.obs_dim)
        return obs

    def step(self, action):
        obs = np.random.rand(self.obs_dim)
        reward = np.sum(np.exp(-np.abs(5. - action))) + (
                self.noise_std * np.random.normal())
        done = True
        return obs, reward, done, {}


def sample_trajectory(env, policy, max_episode_length):
    """Samples one trajectories using the given policy.

    Parameters
    ==========
    env: A gym environment.
    policy: An instance of Policy.
    max_episode_length: Cap on max length for each episode.

    Returns
    =======
    A  dictionary, each dictionary maps keys `observation`, `action`, `reward`,
    `next_observation`, `terminal` to numpy arrays of size episode length.
    """
    # initialize env for the beginning of a new rollout
    ob = env.reset()
    obs, acs, rewards, next_obs, terminals = [], [], [], [], []
    steps = 0
    while True:
        # use the most recent ob to decide what to do
        obs.append(ob)
        action_pd = policy(torch.Tensor(ob[None, :]))
        ac = policy.sample(action_pd).numpy().flatten()
        acs.append(ac)
        # take that action and record results
        ob, rew, done, _ = env.step(ac)
        # record result of taking that action
        steps += 1
        next_obs.append(ob)
        rewards.append(rew)
        # End the rollout if the rollout ended
        # Note that the rollout can end due to done, or due to max_episode_length
        if done or steps >= max_episode_length:
            rollout_done = 1
        else:
            rollout_done = 0
        terminals.append(rollout_done)
        if rollout_done:
            break
    return wrap_trajectory(obs, acs, rewards, next_obs, terminals)


def wrap_trajectory(obs, acs, rewards, next_obs, terminals):
    return {"observation" : np.array(obs, dtype=np.float32),
            "reward" : np.array(rewards, dtype=np.float32),
            "action" : np.array(acs, dtype=np.float32),
            "next_observation": np.array(next_obs, dtype=np.float32),
            "terminal": np.array(terminals, dtype=np.float32)}


def sample_trajectories_by_batch_size(env, policy, min_timesteps_per_batch, max_episode_length):
    """Samples multiple trajectories using the given policy to achieve total number of steps.

    Parameters
    ==========
    env: A gym environment.
    policy: An instance of Policy.
    min_timesteps_per_batch: Desired number of timesteps in all trajectories combined.
    max_episode_length: Cap on max length for each episode.

    Returns
    =======
    A list of n dictionaries, each dictionary maps keys `observation`, `action`,
    `reward`, `next_observation`, `terminal` to numpy arrays of size episode length.
    """
    timesteps_this_batch = 0
    trajectories = []
    while timesteps_this_batch < min_timesteps_per_batch:
        trajectory = sample_trajectory(env, policy, max_episode_length)
        trajectories.append(trajectory)
        timesteps_this_batch += len(trajectory['action'])
    return trajectories, timesteps_this_batch


def get_reward_to_go(rewards, gamma=0.99):
    """Helper function to compute discounted reward to go.

    Parameters
    ==========
    rewards: A list of rewards {r_0, r_1, ..., r_t', ... r_{T-1}} from a single
    rollout of length T.

    Returns
    =======
    A numpy array where the entry at index t is sum_{t'=t}^{T-1} gamma^(t'-t) * r_{t'}.
    """
    all_discounted_cumsums = []
    # for loop over steps (t) of the given rollout
    for start_time_index in range(len(rewards)):
        indices = np.arange(start_time_index, len(rewards))
        discounts = gamma**(indices - start_time_index)
        all_discounted_cumsums.append(sum(discounts * rewards[start_time_index:]))
    return np.array(all_discounted_cumsums)
 

def get_obs_acs_rewards_advs(trajectories):
    obs = np.concatenate([tau["observation"] for tau in trajectories], axis=0)
    acs = np.concatenate([tau["action"] for tau in trajectories], axis=0)
    rewards = np.concatenate([tau["reward"] for tau in trajectories])
    reward_to_go = np.concatenate(
        [get_reward_to_go(tau["reward"]) for tau in trajectories])
    advs = (reward_to_go - np.mean(reward_to_go)) / (np.std(reward_to_go) + 1e-8)
    return obs, acs, rewards, advs


def step(env, policy, params):
    trajs, steps_in_trajs = sample_trajectories_by_batch_size(
        env, policy, params.batch_size, params.max_episode_len)
    obs, acs, rewards, advs = get_obs_acs_rewards_advs(trajs)
    obs = torch.Tensor(obs)
    acs = torch.Tensor(acs)
    advs = torch.Tensor(advs)
    with torch.no_grad():
        old_pds = policy(obs)
        old_log_ps = policy.get_loglikelihood(old_pds, acs)
    loss = ppo_step(obs, acs, old_log_ps, None, None, None, advs, policy, params, None, None)
    return_dict = {
        'loss': loss.detach().item(),
        'mean_reward': np.mean(rewards)
    }
    if isinstance(old_pds, tuple):
        for i, pd_param in enumerate(old_pds):
            return_dict[f'pd_param_{i}_mean'] = np.mean(pd_param.numpy())
    else:
        for i in range(old_pds.shape[1]):
            return_dict[f'pd_param_{i}_mean'] = np.mean(old_pds[:, i].numpy())
    return return_dict


def train(env, policy, params, n_steps):
    df = pd.DataFrame()
    for i in range(n_steps):
        return_dict = step(env, policy, params)
        return_dict['iter'] = i
        df = df.append(return_dict, ignore_index=True)
    return df


def compare(policy_type, action_dim, n_steps=20, repeats=10, seed=0, noise_std=0.1,
        **kwargs):
    np.random.seed(seed)
    # Handle different policy classes.
    policy_type = policy_type.lower()
    policy_cls_map = {
            'gaussian': CtsPolicy, 'beta': CtsBetaPolicy, 'discrete': DiscPolicy}
    policy_cls = policy_cls_map[policy_type]
    env = SimpleEnv(action_dim=action_dim, noise_std=noise_std,
            is_discrete=(policy_type == 'discrete'))
    base_params.update(**kwargs)
    data = None
    for i, name in enumerate(comparisons.keys()):
        params = base_params.copy()
        params.update(comparisons[name])
        for j in range(repeats):
            policy = policy_cls(env.obs_dim, env.action_dim, init='xavier',
                    hidden_sizes=[env.action_dim])
            params['policy_adam'] = torch.optim.Adam(policy.parameters(),
                    lr=params['lr'])
            data_this_run = train(env, policy, Parameters(params), n_steps)
            data_this_run['method'] = name
            if data is None:
                data = data_this_run
            else:
                data = pd.concat([data, data_this_run])
    return data
