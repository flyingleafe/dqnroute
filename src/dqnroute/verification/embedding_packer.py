from collections import OrderedDict
from typing import *

import torch

from ..utils import AgentId
from .router_graph import RouterGraph

class EmbeddingPacker:
    """
    This class establishes a mapping between node-assigned embeddings and their concatenation
    as a single vector.
    """
    
    def __init__(self, g: RouterGraph, sink: AgentId, sink_embedding: torch.Tensor, other_nodes: List[AgentId]):
        self._stored_embeddings = OrderedDict({sink: sink_embedding})
        for node_key in other_nodes:
            self._stored_embeddings[node_key], _, _ = g.node_to_embeddings(node_key, sink)
        self._emb_dim = sink_embedding.flatten().shape[0]
        # the indices of all nodes in this embedding storage:
        self._node_key_to_index = {key: i for i, key in enumerate(self._stored_embeddings.keys())}
        assert self._node_key_to_index[sink] == 0

    def pack(self, embedding_dict: OrderedDict) -> torch.Tensor:
        return torch.cat(tuple(embedding_dict.values())).flatten()

    def unpack(self, embedding_vector: torch.Tensor) -> OrderedDict:
        embedding_dict = OrderedDict()
        for i, (key, value) in enumerate(self._stored_embeddings.items()):
            embedding_dict[key] = embedding_vector[i*self._emb_dim:(i + 1)*self._emb_dim].reshape(1, self._emb_dim)
        return embedding_dict
    
    def initial_vector(self) -> torch.Tensor:
        return self.pack(self._stored_embeddings)
    
    def number_of_embeddings(self) -> int:
        return len(self._stored_embeddings)
    
    def node_key_to_index(self, key: AgentId) -> int:
        return self._node_key_to_index[key]