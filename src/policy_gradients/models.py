import torch.nn as nn
import math
import torch as ch
import torch.nn.functional as F
from .torch_utils import *
from .cts_policy import CtsPolicy
from .cts_beta_policy import CtsBetaPolicy
from .disc_policy import DiscPolicy
from .discretized_cts_policy import DiscretizedCtsPolicy

'''
Neural network models for estimating value and policy functions
Contains:
- Initialization utilities
- Value Network(s)
- Policy Network(s)
- Retrieval Function
'''
            

########################
### INITIALIZATION UTILITY FUNCTIONS:
# Generic Value network, Value network MLP
########################

class ValueDenseNet(nn.Module):
    '''
    An example value network, with support for arbitrarily many
    fully connected hidden layers (by default 2 * 128-neuron layers),
    maps a state of size (state_dim) -> a scalar value.
    '''
    def __init__(self, state_dim, init, init_scale=1.0, hidden_sizes=(64, 64)):
        '''
        Initializes the value network.
        Inputs:
        - state_dim, the input dimension of the network (i.e dimension of state)
        - hidden_sizes, an iterable of integers, each of which represents the size
        of a hidden layer in the neural network.
        Returns: Initialized Value network
        '''
        super().__init__()
        self.activation = ACTIVATION()
        self.affine_layers = nn.ModuleList()

        prev = state_dim
        for h in hidden_sizes:
            l = nn.Linear(prev, h)
            initialize_weights(l, init, scale=STD)
            self.affine_layers.append(l)
            prev = h

        self.final = nn.Linear(prev, 1)
        initialize_weights(self.final, init, scale=init_scale)

    def forward(self, x):
        '''
        Performs inference using the value network.
        Inputs:
        - x, the state passed in from the agent
        Returns:
        - The scalar (float) value of that state, as estimated by the net
        '''
        for affine in self.affine_layers:
            x = self.activation(affine(x))
        value = self.final(x)
        return value

    def get_value(self, x):
        return self(x)

########################
### POLICY NETWORKS
# Discrete and Continuous Policy Examples
########################

'''
A policy network can be any class which is initialized 
with a state_dim and action_dim, as well as optional named arguments.
Must provide:
- A __call__ override (or forward, for nn.Module): 
    * returns a tensor parameterizing a distribution, given a 
    BATCH_SIZE x state_dim tensor representing shape
- A function calc_kl(p, q): 
    * takes in two batches tensors which parameterize probability 
    distributions (of the same form as the output from __call__), 
    and returns the KL(p||q) tensor of length BATCH_SIZE
- A function entropies(p):
    * takes in a batch of tensors parameterizing distributions in 
    the same way and returns the entropy of each element in the 
    batch as a tensor
- A function sample(p): 
    * takes in a batch of tensors parameterizing distributions in
    the same way as above and returns a batch of actions to be 
    performed
- A function get_likelihoods(p, actions):
    * takes in a batch of parameterizing tensors (as above) and an 
    equal-length batch of actions, and returns a batch of probabilities
    indicating how likely each action was according to p.
'''

## Retrieving networks
# Make sure to add newly created networks to these dictionaries!

POLICY_NETS = {
    "DiscPolicy": DiscPolicy,
    "CtsPolicy": CtsPolicy,
    "CtsBetaPolicy": CtsBetaPolicy,
    "DiscretizedCtsPolicy": DiscretizedCtsPolicy,
}

VALUE_NETS = {
    "ValueNet": ValueDenseNet,
}

def policy_net_with_name(name):
    return POLICY_NETS[name]

def value_net_with_name(name):
    return VALUE_NETS[name]

