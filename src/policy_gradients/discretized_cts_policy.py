import numpy as np
import torch.nn as nn
import math
import torch as ch
import torch.nn.functional as F
from .torch_utils import *
from .disc_policy import DiscPolicy


class DiscretizedCtsPolicy(DiscPolicy):
    '''
    A discrete policy to discretize continuous environment.
    '''
    def __init__(self, state_dim, action_dim, init, hidden_sizes=HIDDEN_SIZES,
            time_in_state=False, share_weights=False, action_space_low=None,
            action_space_high=None, n_actions_per_dim=10, **unused_args):

        self.n_actions = n_actions_per_dim * action_dim
        super(DiscretizedCtsPolicy, self).__init__(state_dim, self.n_actions,
                init, hidden_sizes, time_in_state, share_weights)
        self.action_dim = action_dim
        self.n_actions_per_dim = n_actions_per_dim
        self.discrete = False

        assert action_space_low is not None
        assert action_space_high is not None
        assert np.all(-np.inf < action_space_low)
        assert np.all(np.inf > action_space_high)
        self.action_space_low = action_space_low
        self.action_space_high = action_space_high

    def _logits_to_probs(self, logits):
        '''
        Converts logits in shape [batch_size, n_actions_per_dim * action_dim] to
        probabilities by applying softmax for each action dim.
        '''
        probs = []
        logits = ch.reshape(logits, (-1, self.action_dim, self.n_actions_per_dim))
        for i in range(self.action_dim):
            probs.append(F.softmax(logits[:, i, :], dim=-1))
        return ch.stack(probs, dim=1)

    def _disc_action_to_cts(self, actions):
        # First map to 0 to 1
        cts_actions_normalized = actions.float() / (self.n_actions_per_dim - 1)
        return cts_actions_normalized * (self.action_space_high -
                self.action_space_low) + self.action_space_low

    def _cts_action_to_disc(self, actions):
        # First map back to 0 to 1.
        cts_actions_normalized = (actions - self.action_space_low) / (
                self.action_space_high - self.action_space_low)
        return ch.round(cts_actions_normalized * (self.n_actions_per_dim -
            1)).long()

    def forward(self, x):
        '''
        Outputs the categorical distribution (via softmax)
        by feeding the state through the neural network
        '''
        # If the time is in the state, discard it
        if self.time_in_state:
            x = x[:,:-1]
        for affine in self.affine_layers:
            x = self.activation(affine(x))
        
        probs = self._logits_to_probs(self.final(x))
        return probs

    def calc_kl(self, p, q, get_mean=True, **unused_args):
        assert shape_equal([-1, self.action_dim, self.n_actions_per_dim], p, q)
        kl_per_dim = []
        for i in range(self.action_dim):
            kl_per_dim.append(super(DiscretizedCtsPolicy, self).calc_kl(
                    p[:, i, :], q[:, i, :], get_mean))
        return ch.sum(ch.stack(kl_per_dim), dim=0)

    def entropies(self, p):
        assert shape_equal([-1, self.action_dim, self.n_actions_per_dim], p)
        entropy_per_dim = []
        for i in range(self.action_dim):
            entropy_per_dim.append(
                    super(DiscretizedCtsPolicy, self).entropies(p[:, i, :]))
        return ch.sum(ch.stack(entropy_per_dim), dim=0)

    def get_loglikelihood(self, p, actions):
        assert shape_equal([-1, self.action_dim, self.n_actions_per_dim], p)
        assert shape_equal([-1, self.action_dim], actions)
        disc_actions = self._cts_action_to_disc(actions)
        assert shape_equal([-1, self.action_dim], disc_actions.detach().numpy())
        ll_per_dim = []
        for i in range(self.action_dim):
            ll_per_dim.append(super(DiscretizedCtsPolicy, self).get_loglikelihood(
                p[:, i, :], disc_actions[:, i]))
        return ch.sum(ch.stack(ll_per_dim), dim=0)

    def sample(self, probs):
        assert shape_equal([-1, self.action_dim, self.n_actions_per_dim], probs)
        action_per_dim = []
        for i in range(self.action_dim):
            action_per_dim.append(super(DiscretizedCtsPolicy, self).sample(
                probs[:, i, :]))
        disc_actions = ch.stack(action_per_dim, dim=1)
        return self._disc_action_to_cts(disc_actions)


