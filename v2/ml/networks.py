import torch
import torch.nn as nn

from typing import List

from base import TorchModel
from . import get_activation


class LinearBlock(nn.Module):

    def __init__(
            self,
            input_dim: int,
            intermediate_dim: int,
            output_dim: int = None,
            activation: nn.Module = nn.ReLU(),
            use_layernorm: bool = False,
            use_dropout: bool = False,
            use_residual: bool = False,
            dropout: float = 0.0,
            eps: float = 1e-12,
            initializer_range: float = 0.02
    ):
        super().__init__()
        self._ff1 = nn.Linear(input_dim, intermediate_dim)

        self._ff2 = nn.Identity()
        if output_dim is not None:
            self._ff2 = nn.Linear(intermediate_dim, output_dim)

        self._activation = activation

        self._layernorm = nn.Identity()
        if use_layernorm:
            self._layernorm = nn.LayerNorm(intermediate_dim or output_dim, eps=eps)

        self._dropout = nn.Identity()
        if use_dropout:
            self._dropout = nn.Dropout(p=dropout)

        self._init_weights(self._ff1, initializer_range=initializer_range, use_xavier=True)
        if isinstance(self._ff2, nn.Linear):
            self._init_weights(self._ff2, initializer_range=initializer_range, use_xavier=True)

        self._use_residual = use_residual

    @staticmethod
    def _init_weights(layer, initializer_range=0.02, use_xavier=False):
        if use_xavier:
            nn.init.xavier_uniform_(layer.weight)
        else:
            nn.init.trunc_normal_(
                layer.weight,
                std=initializer_range,
                a=-2 * initializer_range,
                b=2 * initializer_range
            )
        nn.init.zeros_(layer.bias)

    def forward(self, embeddings):
        if self._use_residual:
            embeddings = self._layernorm(self._dropout(self._ff2(self._activation(self._ff1(embeddings)))) + embeddings)
        else:
            embeddings = self._layernorm(self._dropout(self._ff2(self._activation(self._ff1(embeddings)))))

        return embeddings


class FeedForwardModel(TorchModel, config_name='feed_forward'):
    """
    Simple feed-forward network with fully connected layers
    """

    def __init__(
            self,
            prefix: str,
            hidden_dims: List[int],
            activation: str = 'relu',
            input_dim: int = None,
            output_dim: int = None,
            output_prefix: str = None,
            dropout=0.0,
            initializer_range=0.02,
            eps=1e-12
    ):
        super().__init__()
        self._prefix = prefix
        self._output_prefix = output_prefix or output_prefix

        self.hidden_dims = hidden_dims
        self._activation = get_activation(activation)

        self._input_dim = input_dim
        self._output_dim = output_dim

        self._input_projector = nn.Identity()
        if input_dim is not None:
            self._input_projector = nn.Linear(input_dim, hidden_dims[0])

        self._layers = nn.Sequential(*[
            LinearBlock(
                input_dim=hidden_dims[i],
                intermediate_dim=hidden_dims[i + 1],
                output_dim=None,
                eps=eps,
                dropout=dropout,
                initializer_range=initializer_range
            ) for i in range(len(hidden_dims) - 1)
        ])

        self._output_projector = nn.Identity()
        if output_dim is not None:
            self._output_projector = nn.Linear(hidden_dims[-1], output_dim)

        self._init_weights(initializer_range=initializer_range, use_xavier=True)

    def _init_weights(self, initializer_range=0.02, use_xavier=False):
        for layer in [self._input_projector, self._output_projector]:
            if isinstance(layer, nn.Linear):
                if use_xavier:
                    nn.init.xavier_uniform_(layer.weight)
                else:
                    nn.init.trunc_normal_(
                        layer.weight,
                        std=initializer_range,
                        a=-2 * initializer_range,
                        b=2 * initializer_range
                    )
                nn.init.zeros_(layer.bias)

    def forward(self, inputs):
        embeddings = inputs[self._prefix]

        embeddings = self._input_projector(embeddings)
        embeddings = self._layers(embeddings)
        embeddings = self._output_projector(embeddings)

        inputs[self._output_prefix] = embeddings

        return inputs
