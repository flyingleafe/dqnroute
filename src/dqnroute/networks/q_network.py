import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils import *
from ..constants import INFTY
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

class QNetwork(SaveableModel):
    """
    Simple Q-network with one-hot encoded inputs
    """

    def __init__(self, n, layers, activation, additional_inputs=[],
                 embedding_dim=None, scope='', **kwargs):
        super().__init__()
        self.graph_size = n
        self.add_inputs = _transform_add_inputs(n, additional_inputs)
        self.uses_embedding = embedding_dim is not None

        input_dim = sum([d for (_, d) in self.add_inputs])
        if not self.uses_embedding:
            input_dim += 3 * n
        else:
            input_dim += 3 * embedding_dim

        self._scope = scope if len(scope) > 0 else None
        self._label = 'qnetwork-oneinp{}_{}_{}_{}_{}_{}'.format(
            '-emb-{}'.format(embedding_dim) if self.uses_embedding else '',
            input_dim,
            '-'.join(map(str, layers)),
            n,
            activation,
            '_'.join(map(lambda p: p[0]+'-'+str(p[1]), self.add_inputs)))

        self.ff_net = FFNetwork(input_dim, 1, layers=layers, activation=activation)

    def forward(self, addr, dst, neighbour, *others):
        if self.uses_embedding:
            addr_ = atleast_dim(addr, 2)
            dst_ = atleast_dim(dst, 2)
            neighbour_ = atleast_dim(neighbour, 2)
        else:
            addr_ = one_hot(atleast_dim(addr, 1), self.graph_size)
            dst_ = one_hot(atleast_dim(dst, 1), self.graph_size)
            neighbour_ = one_hot(atleast_dim(neighbour, 1), self.graph_size)

        input_tensors = [addr_, dst_, neighbour_]

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
        return self.ff_net(input)
