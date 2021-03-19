import os
import torch
import torch.nn as nn
import torch.optim as optim

from typing import List
from functools import partial
from ..constants import TORCH_MODELS_DIR

def get_activation(name):
    if type(name) != str:
        return name
    if name == 'relu':
        return nn.ReLU()
    elif name == 'tanh':
        return nn.Tanh()
    elif name == 'sigmoid':
        return nn.Sigmoid()
    else:
        raise Exception('Unknown activation function: ' + name)

def get_optimizer(name, params={}):
    if isinstance(name, optim.Optimizer):
        return name
    if name == 'rmsprop':
        return partial(optim.RMSprop, **dict(params, lr=0.001))
    elif name == 'adam':
        return partial(optim.Adam, **params)
    elif name == 'adadelta':
        return partial(optim.Adadelta, **params)
    elif name == 'adagrad':
        return partial(optim.Adagrad, **dict(params, lr=0.001))
    else:
        raise Exception('Invalid optimizer: ' + str(name))

def get_distance_function(name):
    if type(name) != str:
        return name
    if name == 'euclid':
        return euclidean_distance
    elif name == 'linear':
        pass
    elif name == 'cosine':
        pass
    else:
        raise Exception(f'Invalid distance function: {str(name)}')

def atleast_dim(x: torch.Tensor, dim: int, axis=0) -> torch.Tensor:
    while x.dim() < dim:
        x = x.unsqueeze(axis)
    return x

def one_hot(indices, dim) -> torch.Tensor:
    """
    Creates a one-hot tensor from a tensor of integers.
    indices.size() should be [seq,batch] or [batch,]
    result size() would be [seq,batch,dim] or [batch,dim]
    """
    out = torch.zeros(indices.size()+torch.Size([dim]))
    d = len(indices.size())
    return out.scatter_(d, indices.unsqueeze(d).to(dtype=torch.int64), 1)

def xavier_init(m: nn.Module):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)

class FFNetwork(nn.Sequential):
    """
    Simple feed-forward network with fully connected layers
    """

    def __init__(self, input_dim: int, output_dim: int, layers: List[int], activation='relu'):
        super().__init__()
        act_module = get_activation(activation)

        prev_dim = input_dim
        for i, layer in enumerate(layers):
            if type(layer) == int:
                lsize = layer
                self.add_module(f'fc{i + 1}', nn.Linear(prev_dim, lsize))
                self.add_module(f'activation{i + 1}', act_module)
                prev_dim = lsize
            elif layer == 'dropout':
                self.add_module(f'dropout_{i + 1}', nn.Dropout())

        self.add_module('output', nn.Linear(prev_dim, output_dim))

class SaveableModel(nn.Module):
    """
    Mixin which provides `save` and `restore`
    methods for (de)serializing the model.
    """
    def _savedir(self):
        dir = TORCH_MODELS_DIR
        if self._scope is not None:
            dir += '/' + self._scope
        return dir

    def _savepath(self):
        return self._savedir() + '/' + self._label

    def save(self):
        os.makedirs(self._savedir(), exist_ok=True)
        return torch.save(self.state_dict(), self._savepath())

    def restore(self):
        return self.load_state_dict(torch.load(self._savepath()))


# distance functions
def euclidean_distance(allowed_neighbours: torch.Tensor, existing_state: torch.Tensor):
    sum = torch.sum((allowed_neighbours - existing_state) ** 2, dim=1)
    return torch.sqrt(sum)


def linear_distance(existing_state: torch.Tensor, predicted_state: torch.Tensor):
    pass  # TODO Implement


def cosine_distance(existing_state, predicted_state):
    pass  # TODO Implement