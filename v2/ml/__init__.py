import torch
import torch.nn as nn
import torch.optim as optim

from functools import partial


def get_activation(name: str):  # TODO [Vladimir Baikalov]: implement via Meta-Classes (maybe)
    if type(name) != str:
        return name
    if name == 'relu':
        return nn.ReLU()
    elif name == 'tanh':
        return nn.Tanh()
    elif name == 'sigmoid':
        return nn.Sigmoid()
    else:
        raise ValueError(f'Invalid activation function: {name}')


def get_optimizer(name: str, params=None):  # TODO [Vladimir Baikalov]: implement via Meta-Classes  (and enrich scheme)
    params = params or {}

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
        raise ValueError(f'Invalid optimizer: {name}')


def add_dim(x: torch.Tensor, dim_size: int, add_first: bool) -> torch.Tensor:
    if x.dim() > dim_size:
        raise ValueError(f'Input tensor already has {x.dim()} dimensions. Required dimensions num: {dim_size}')

    while x.dim() < dim_size:
        x = x.unsqueeze(0 if add_first else -1)

    return x
