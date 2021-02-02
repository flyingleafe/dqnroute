from collections import OrderedDict
from typing import *

import torch

from ..utils import AgentId
from .ml_util import Util
from .router_graph import RouterGraph


class EmbeddingPacker:
    """
    This class establishes a mapping between node-assigned embeddings and their concatenation
    as a single vector.
    """
    
    def __init__(self, g: RouterGraph, sink: AgentId, sink_embedding: torch.Tensor, other_nodes: List[AgentId]):
        self._stored_embeddings = OrderedDict({sink: sink_embedding})
        
        other_nodes_set = set(other_nodes)
        other_nodes_set.add(sink)
        
        for node_key in other_nodes:
            # for each diverter with two reachable successors, add all these nodes
            if node_key[0] == "diverter":
                neighbor_keys = g.get_out_nodes(node_key)
                if neighbor_keys[0] in other_nodes_set and neighbor_keys[1] in other_nodes_set:
                    for new_node in (node_key, *neighbor_keys):
                        self._stored_embeddings[new_node], _, _ = g.node_to_embeddings(new_node, sink)
        
        self._emb_dim = sink_embedding.flatten().shape[0]
        # the indices of all nodes in this embedding storage:
        self._node_key_to_index = {key: i for i, key in enumerate(self._stored_embeddings.keys())}
        assert self._node_key_to_index[sink] == 0
        self._g = g
        self._sink = sink
        print(f"  Actually used embeddings: {sorted(self._stored_embeddings.keys())}, "
              f"total number = {len(self._stored_embeddings)}")

    def pack(self, embedding_dict: OrderedDict) -> torch.Tensor:
        return torch.cat(tuple(embedding_dict.values())).flatten()

    def unpack(self, embedding_vector: torch.Tensor) -> OrderedDict:
        embedding_dict = OrderedDict()
        for i, (key, value) in enumerate(self._stored_embeddings.items()):
            i1 = self._emb_dim * i
            i2 = self._emb_dim * (i + 1)
            embedding_dict[key] = embedding_vector[i1:i2].reshape(1, self._emb_dim)
        return embedding_dict
    
    def initial_vector(self) -> torch.Tensor:
        return self.pack(self._stored_embeddings)
    
    def number_of_embeddings(self) -> int:
        return len(self._stored_embeddings)
    
    def node_key_to_index(self, key: AgentId) -> int:
        return self._node_key_to_index[key]
    
    def compute_objective(self, embedding_dict: OrderedDict, nontrivial_diverters: List[AgentId],
                          lambdified_objective: Callable, softmax_temperature: float,
                          probability_smoothing: float) -> torch.Tensor:
        ps = []
        sink_embeddings = embedding_dict[self._sink].repeat(2, 1)
        for diverter in nontrivial_diverters:
            diverter_embeddings = embedding_dict[diverter].repeat(2, 1)
            _, current_neighbors, _ = self._g.node_to_embeddings(diverter, self._sink)
            neighbor_embeddings = torch.cat([embedding_dict[current_neighbor]
                                             for current_neighbor in current_neighbors])
            q_values = self._g.q_forward(diverter_embeddings, sink_embeddings, neighbor_embeddings).flatten()
            ps += [Util.q_values_to_first_probability(q_values, softmax_temperature, probability_smoothing)]
        return lambdified_objective(*ps), ps