import torch.nn as nn
import math
import torch as ch
import torch.nn.functional as F
from .torch_utils import *


class CtsPolicy(nn.Module):
    '''
    A continuous policy using a fully connected neural network.
    The parameterizing tensor is a mean and standard deviation vector, 
    which parameterize a gaussian distribution.
    '''
    def __init__(self, state_dim, action_dim, init, hidden_sizes=HIDDEN_SIZES,
                 time_in_state=False, share_weights=False,
                 action_space_low=None, action_space_high=None,
                 adjust_init_std=False):
        super().__init__()
        self.activation = ACTIVATION()
        self.action_dim = action_dim
        self.discrete = False
        self.time_in_state = time_in_state

        self.affine_layers = nn.ModuleList()
        prev_size = state_dim
        for i in hidden_sizes:
            lin = nn.Linear(prev_size, i)
            initialize_weights(lin, init)
            self.affine_layers.append(lin)
            prev_size = i

        self.final_mean = nn.Linear(prev_size, action_dim)
        initialize_weights(self.final_mean, init, scale=0.01)
        
        # For the case where we want to share parameters 
        # between the policy and value networks
        self.share_weights = share_weights
        if share_weights:
            if time_in_state:
                self.final_value = nn.Linear(prev_size + 1, 1)
            else:
                self.final_value = nn.Linear(prev_size, 1)

            initialize_weights(self.final_value, init, scale=1.0)

        if adjust_init_std: 
            assert action_space_low is not None
            assert action_space_high is not None
            assert np.all(-np.inf < action_space_low)
            assert np.all(np.inf > action_space_high)
            # symmetric action spaces
            # initializing final mean to be around 0 only makes sense if symmetric
            assert (np.mean(action_space_low + action_space_high).round(2) == 0.0).all()
            # initialize std such that high and low are at ~ 2 STD
            stdev_init = (action_space_high - action_space_low) / 4
            log_stdev_init = ch.Tensor(np.log(stdev_init))
        else:
            log_stdev_init = ch.Tensor(ch.zeros(action_dim))
        self.log_stdev = ch.nn.Parameter(log_stdev_init)

    def forward(self, x):
        # If the time is in the state, discard it
        if self.time_in_state:
            x = x[:,:-1]
        for affine in self.affine_layers:
            x = self.activation(affine(x))
        
        means = self.final_mean(x)
        std = ch.exp(self.log_stdev)

        return means, std 

    def get_value(self, x):
        assert self.share_weights, "Must be sharing weights to use get_value"

        # If the time is in the state, discard it
        t = None
        if self.time_in_state:
            t = x[...,-1:]
            x = x[...,:-1]

        for affine in self.affine_layers:
            x = self.activation(affine(x))

        if self.time_in_state:
            return self.final_value(ch.cat((x, t), -1))
        else:
            return self.final_value(x)

    def sample(self, p):
        '''
        Given prob dist (mean, var), return: actions sampled from p_i, and their
        probabilities. p is tuple (means, var). means shape 
        (batch_size, action_space), var (action_space,), here are batch_size many
        prboability distributions you're sampling from

        Returns tuple (actions, probs):
        - actions: shape (batch_size, action_dim)
        - probs: shape (batch_size, action_dim)
        '''
        means, std = p
        return (means + ch.randn_like(means)*std).detach()

    def get_loglikelihood(self, p, actions):
        try:    
            mean, std = p
            nll =  0.5 * ((actions - mean).pow(2) / std.pow(2)).sum(-1) \
                   + 0.5 * np.log(2.0 * np.pi) * actions.shape[-1] \
                   + self.log_stdev.sum(-1)
            return -nll
        except Exception as e:
            raise ValueError("Numerical error")

    def calc_kl(self, p, q, npg_approx=False, get_mean=True):
        '''
        Get the expected KL distance between two sets of gaussians over states -
        gaussians p and q where p and q are each tuples (mean, var)
        - In other words calculates E KL(p||q): E[sum p(x) log(p(x)/q(x))]
        - From https://stats.stackexchange.com/a/60699
        '''
        p_mean, p_std = p
        q_mean, q_std = q
        p_var = p_std.pow(2) + 1e-10
        q_var = q_std.pow(2) + 1e-10
        assert shape_equal([-1, self.action_dim], p_mean, q_mean)
        assert shape_equal([self.action_dim], p_var, q_var)

        d = q_mean.shape[1]
        # Add 1e-10 to variances to avoid nans.
        logdetp = log_determinant(p_var)
        logdetq = log_determinant(q_var)
        diff = q_mean - p_mean

        log_quot_frac = logdetq - logdetp
        tr = (p_var / q_var).sum()
        quadratic = ((diff / q_var) * diff).sum(dim=1)

        if npg_approx:
            kl_sum = 0.5 * quadratic + 0.25 * (p_var / q_var - 1.).pow(2).sum()
        else:
            kl_sum = 0.5 * (log_quot_frac - d + tr + quadratic)
        assert kl_sum.shape == (p_mean.shape[0],)
        if get_mean:
            return kl_sum.mean()
        return kl_sum

    def entropies(self, p):
        '''
        Get entropies over the probability distributions given by p
        p_i = (mean, var), p mean is shape (batch_size, action_space),
        p var is shape (action_space,)
        '''
        _, std = p
        var = std.pow(2) + 1e-10
        # Add 1e-10 to variance to avoid nans.
        logdetp = log_determinant(var)
        d = var.shape[0]
        entropies = 0.5 * (logdetp + d * (1. + math.log(2 * math.pi)))
        return entropies


