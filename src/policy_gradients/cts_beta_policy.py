import torch.nn as nn
import math
import torch as ch
import torch.nn.functional as F
from .torch_utils import *


class CtsBetaPolicy(nn.Module):
    '''
    A continuous policy using a fully connected neural network.
    The parameterizing tensor is an alpha and beta vector
    which parameterize a beta distribution.
    '''
    def __init__(self, state_dim, action_dim, init, hidden_sizes=HIDDEN_SIZES,
                 time_in_state=False, share_weights=False,
                 action_space_low=None, action_space_high=None, **unused_args):
        super().__init__()
        assert action_space_low is not None
        assert action_space_high is not None
        assert np.all(-np.inf < action_space_low)
        assert np.all(np.inf > action_space_high)
        self.action_space_low = action_space_low
        self.action_space_high = action_space_high

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

        self.alpha_pre_softplus = nn.Linear(prev_size, action_dim)
        initialize_weights(self.alpha_pre_softplus, init, scale=0.01)
        self.beta_pre_softplus = nn.Linear(prev_size, action_dim)
        initialize_weights(self.beta_pre_softplus, init, scale=0.01)
        self.softplus = ch.nn.Softplus()
        
        # For the case where we want to share parameters 
        # between the policy and value networks
        self.share_weights = share_weights
        if share_weights:
            if time_in_state:
                self.final_value = nn.Linear(prev_size + 1, 1)
            else:
                self.final_value = nn.Linear(prev_size, 1)

            initialize_weights(self.final_value, init, scale=1.0)

    def forward(self, x):
        # If the time is in the state, discard it
        if self.time_in_state:
            x = x[:,:-1]
        for affine in self.affine_layers:
            x = self.activation(affine(x))

        # Use alpha and beta >= 1 according to [Chou et. al, 2017]
        alpha = ch.add(self.softplus(self.alpha_pre_softplus(x)), 1.)
        beta = ch.add(self.softplus(self.beta_pre_softplus(x)), 1.)
        return alpha, beta

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

    def scale_by_action_bounds(self, beta_dist_samples):
        # Scale [0, 1] back to action space.
        return beta_dist_samples * (self.action_space_high -
                self.action_space_low) + self.action_space_low

    def inv_scale_by_action_bounds(self, actions):
        # Scale action space to [0, 1].
        return (actions - self.action_space_low) / (self.action_space_high -
                self.action_space_low)


    def sample(self, p):
        '''
        Given prob dist (alpha, beta), return: actions sampled from p_i, and their
        probabilities. p is tuple (alpha, beta). means shape 
        (batch_size, action_space), var (action_space,), here are batch_size many
        prboability distributions you're sampling from

        Returns tuple (actions, probs):
        - actions: shape (batch_size, action_dim)
        - probs: shape (batch_size, action_dim)
        '''
        alpha, beta = p
        dist = ch.distributions.beta.Beta(alpha, beta)
        samples = dist.sample()
        assert shape_equal([-1, self.action_dim], samples, alpha, beta)
        return self.scale_by_action_bounds(samples)

    def get_loglikelihood(self, p, actions):
        alpha, beta = p
        dist = ch.distributions.beta.Beta(alpha, beta)
        log_probs = dist.log_prob(self.inv_scale_by_action_bounds(actions))
        assert shape_equal([-1, self.action_dim], log_probs, alpha, beta)
        return ch.sum(log_probs, dim=1)

    def lbeta(self, alpha, beta):
        '''The log beta function.'''
        return ch.lgamma(alpha) + ch.lgamma(beta) - ch.lgamma(alpha+beta)

    def calc_kl(self, p, q, npg_approx=False, get_mean=True):
        '''
        Get the expected KL distance between beta distributions.
        '''
        assert not npg_approx
        p_alpha, p_beta = p
        q_alpha, q_beta = q
        assert shape_equal([-1, self.action_dim], p_alpha, p_beta, q_alpha,
                q_beta)

        # Expectation of log x under p.
        e_log_x = ch.digamma(p_alpha) - ch.digamma(p_alpha + p_beta)
        # Expectation of log (1-x) under p.
        e_log_1_m_x = ch.digamma(p_beta) - ch.digamma(p_alpha + p_beta)
        kl_per_action_dim = (p_alpha - q_alpha) * e_log_x
        kl_per_action_dim += (p_beta - q_beta) * e_log_1_m_x
        kl_per_action_dim -= self.lbeta(p_alpha, p_beta)
        kl_per_action_dim += self.lbeta(q_alpha, q_beta)
        # By chain rule on KL divergence.
        kl_joint = ch.sum(kl_per_action_dim, dim=1)
        if get_mean:
            return kl_joint.mean()
        return kl_joint

    def entropies(self, p):
        '''
        Get entropies over the probability distributions given by p
        p_i = (alpha, beta), p mean is shape (batch_size, action_space),
        p var is shape (action_space,)
        '''
        alpha, beta = p
        entropies = self.lbeta(alpha, beta)
        entropies -= (alpha - 1) * ch.digamma(alpha)
        entropies -= (beta - 1) * ch.digamma(beta)
        entropies += (alpha + beta - 2) * ch.digamma(alpha + beta)
        return ch.sum(entropies, dim=1)
