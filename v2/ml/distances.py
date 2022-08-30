import torch

from v2.utils import MetaParent


class BaseDistance(metaclass=MetaParent):
    pass


class TorchDistance(BaseDistance, torch.nn.Module):

    def __init__(
            self,
            neighbors_prefix: str,
            state_prefix: str,
            output_prefix: str,
    ):
        super().__init__()
        self._neighbors_prefix = neighbors_prefix
        self._state_prefix = state_prefix
        self._output_prefix = output_prefix

    def forward(self, inputs):
        raise NotImplementedError


class EuclideanDistance(TorchDistance, config_name='euclidean'):

    def __init__(
            self,
            neighbors_prefix: str,
            state_prefix: str,
            output_prefix: str,
            add_dim: bool = False
    ):
        super().__init__(neighbors_prefix, state_prefix, output_prefix)
        self._add_dim = add_dim

    def forward(self, inputs):
        neighbors = inputs[self._neighbors_prefix]  # (batch_size, neighbors_cnt, embedding_dim)
        state = inputs[self._state_prefix]  # (batch_size, embedding_dim)
        state = state.unsqueeze(1)  # (batch_size, 1, embedding_dim)

        distances = torch.sqrt(torch.sum((neighbors - state) ** 2, dim=-1))  # (batch_size, neighbors_cnt)

        inputs[self._output_prefix] = distances  # (batch_size, neighbors_cnt)
        if self._add_dim:
            inputs[self._output_prefix].unsqueeze(-1)  # (batch_size, neighbors_cnt, -1)

        return inputs


class CosineDistance(TorchDistance, config_name='cosine'):

    def __init__(
            self,
            neighbors_prefix: str,
            state_prefix: str,
            output_prefix: str,
            add_dim: bool = False,
            return_similarity: bool = False,
            normalize: bool = True,
            eps: float = 1e-8
    ):
        super().__init__(neighbors_prefix, state_prefix, output_prefix)
        self._add_dim = add_dim
        self._return_similarity = return_similarity
        self._normalize = normalize
        self._eps = eps

    def forward(self, inputs):
        neighbors = inputs[self._neighbors_prefix]  # (batch_size, neighbors_cnt, embedding_dim)
        state = inputs[self._state_prefix]  # (batch_size, embedding_dim)
        state = state.unsqueeze(1)  # (batch_size, 1, embedding_dim)

        distances = torch.cosine_similarity(
            neighbors, state, dim=-1, eps=self._eps
        )  # (batch_size, neighbors_cnt)

        if not self._return_similarity:
            # cosine_similarity = 1 - cosine -> cosine = 1 - cosine_similarity
            distances = 1 - distances  # (batch_size, neighbors_cnt)

        if not self._normalize:
            # cosine = dot_product / (fst_len * snd_len) -> dot_product = cosine * fst_len * snd_len
            neighbors_len = torch.norm(neighbors, dim=-1)  # (batch_size, neighbors_cnt)
            state_len = torch.norm(state, dim=-1)  # (batch_size, 1)
            distances *= neighbors_len
            distances *= state_len

        inputs[self._output_prefix] = distances

        return inputs


class LinearDistance(TorchDistance, config_name='linear'):
    pass  # TODO [Vladimir Baikalov]: implement
