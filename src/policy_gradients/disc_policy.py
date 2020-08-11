import torch.nn as nn
import math
import torch as ch
import torch.nn.functional as F
from .torch_utils import *


class DiscPolicy(nn.Module):
    '''
    A discrete policy using a fully connected neural network.
    The parameterizing tensor is a categorical distribution over actions
    '''
    def __init__(self, state_dim, action_dim, init, hidden_sizes=HIDDEN_SIZES,
            time_in_state=False, share_weights=False, **unused_args):
        '''
        Initializes the network with the state dimensionality and # actions
        Inputs:
        - state_dim, dimensionality of the state vector
        - action_dim, # of possible discrete actions
        - hidden_sizes, an iterable of length #layers,
            hidden_sizes[i] = number of neurons in layer i
        - time_in_state, a boolean indicating whether the time is 
            encoded in the state vector
        '''
        super().__init__()
        self.activation = ACTIVATION()
        self.time_in_state = time_in_state

        self.discrete = True
        self.affine_layers = nn.ModuleList()
        prev_size = state_dim
        for i in hidden_sizes:
            lin = nn.Linear(prev_size, i)
            initialize_weights(lin, init)
            self.affine_layers.append(lin)
            prev_size = i

        self.final = nn.Linear(prev_size, action_dim)

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
        '''
        Outputs the categorical distribution (via softmax)
        by feeding the state through the neural network
        '''
        # If the time is in the state, discard it
        if self.time_in_state:
            x = x[:,:-1]
        for affine in self.affine_layers:
            x = self.activation(affine(x))
        
        probs = F.softmax(self.final(x), dim=-1)
        return probs

    def calc_kl(self, p, q, get_mean=True, **unused_args):
        '''
        Calculates E KL(p||q):
        E[sum p(x) log(p(x)/q(x))]
        Inputs:
        - p, first probability distribution (NUM_SAMPLES, NUM_ACTIONS)
        - q, second probability distribution (NUM_SAMPLES, NUM_ACTIONS)
        - get_mean, whether to return mean or a list
        Returns:
        - Empirical KL from p to q
        '''
        p, q = p.squeeze(), q.squeeze()
        assert shape_equal_cmp(p, q)
        kl = (p * (ch.log(p + 1e-10) - ch.log(q + 1e-10))).sum(-1)
        if get_mean:
            return kl.mean()
        return kl

    def entropies(self, p):
        '''
        p is probs of shape (batch_size, action_space). return mean entropy
        across the batch of states
        '''
        entropies = (p * ch.log(p)).sum(dim=1)
        return entropies

    def get_loglikelihood(self, p, actions):
        '''
        Inputs:
        - p, batch of probability tensors
        - actions, the actions taken
        '''
        try:
            dist = ch.distributions.categorical.Categorical(p)
            actions = actions.squeeze()
            return dist.log_prob(actions)
        except Exception as e:
            print('probs', p.detach().numpy())
            print('actions', actions.detach().numpy())
            print(e)
            raise ValueError("Numerical error")
    
    def sample(self, probs):
        '''
        given probs, return: actions sampled from P(.|s_i), and their
        probabilities
        - s: (batch_size, state_dim)
        Returns actions:
        - actions: shape (batch_size,)
        '''
        dist = ch.distributions.categorical.Categorical(probs)
        actions = dist.sample()
        return actions.long()

    def get_value(self, x):
        # If the time is in the state, discard it
        assert self.share_weights, "Must be sharing weights to use get_value"
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
