import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils import *
from ..constants import TORCH_MODELS_DIR, INFTY
from .common import *

def _transform_add_inputs(n, add_inputs):
    """
    Transforms a config section into internal
    representation
    """
    def _get_dim(inp):
        if inp['tag'] == 'amatrix':
            return n * n
        else:
            return inp.get('dim', n)

    return [(inp['tag'], _get_dim(inp)) for inp in add_inputs]

class QNetwork(nn.Module):

    def __init__(self, n, layers=[64, 64], activation='relu',
                 additional_inputs=[], scope='', **kwargs):
        super().__init__()
        self.graph_size = n
        self.add_inputs = _transform_add_inputs(n, additional_inputs)

        input_dim = 3 * n + sum([d for (_, d) in self.add_inputs])

        self._label = 'qnetwork_{}_{}_{}_{}_{}'.format(
            input_dim,
            '-'.join(map(str, layers)),
            n,
            activation,
            '_'.join(map(lambda p: p[0]+'-'+str(p[1]), self.add_inputs)))

        if len(scope) > 0:
            self._label = scope + '/' + self._label

        self.ff_net = FFNetwork(input_dim, n, layers=layers, activation=activation)

    def forward(self, addr, dst, neighbours, *others):
        addr_ = one_hot(atleast_dim(addr, 1), self.graph_size)
        dst_ = one_hot(atleast_dim(dst, 1), self.graph_size)
        neighbours = atleast_dim(neighbours, 2)

        input_tensors = [addr_, dst_, neighbours]
        for ((tag, dim), inp) in zip(self.add_inputs, others):
            inp = atleast_dim(inp, 2)
            if inp.size()[1] != dim:
                raise Exception('Wrong {} input dimension: expected {}, actual {}'
                                .format(tag, dim, inp.size()[1]))

            if tag == 'amatrix':
                input_tensors.append(torch.flatten(inp, start_dim=1))
            else:
                input_tensors.append(inp)

        input = torch.cat(input_tensors, dim=1)
        output = self.ff_net(input)

        # Mask out unconnected neighbours with -INFTY values
        inf_mask = torch.mul(torch.add(neighbours, -1), INFTY)
        return torch.add(output, inf_mask)

    def _savepath(self):
        return TORCH_MODELS_DIR + '/' + self._label

    def save(self):
        return torch.save(self.state_dict(), self._savepath())

    def restore(self):
        return self.load_state_dict(torch.load(self._savepath()))
