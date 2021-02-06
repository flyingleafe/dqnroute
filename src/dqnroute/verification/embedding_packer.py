from collections import OrderedDict
from typing import *

import torch

from ..utils import AgentId
from .ml_util import Util
from .router_graph import RouterGraph


class EmbeddingPacker:
    """
    This class establishes a mapping between node-assigned embeddings and their concatenation
    as a single vector. This mapping depends on the sink and the source.
    """
    
    def __init__(self, g: RouterGraph, sink: AgentId, sink_embedding: torch.Tensor, other_nodes: List[AgentId]):
        """
        Constructs EmbeddingPacker for the delivery problem from a fixed source to the fixed sink.
        :param g: RouterGraph.
        :param sink: the sink.
        :param sink_embedding: the embedding of the sink.
        :param other_nodes: list of all nodes that belong to any paths between the source and the sink.
        """
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

    def pack(self, embedding_dict: OrderedDict[AgentId, torch.Tensor]) -> torch.Tensor:
        """
        Packs a dictionary of embeddings to a single vector.
        :param embedding_dict: OrderedDict to pack.
        :return: concatenation of all the embeddings in the order of embedding_dict as a PyTorch tensor.
        """
        return torch.cat(tuple(embedding_dict.values())).flatten()

    def unpack(self, embedding_vector: torch.Tensor) -> OrderedDict[AgentId, torch.Tensor]:
        """
        Unpacks a vector to a dictionary of embeddings. This method is the reverse of pack().
        :param embedding_vector: concatenation of all the embeddings produced by pack().
        :return: OrderedDict of all the embeddings.
        """
        embedding_dict = OrderedDict()
        for i, (key, value) in enumerate(self._stored_embeddings.items()):
            i1 = self._emb_dim * i
            i2 = self._emb_dim * (i + 1)
            embedding_dict[key] = embedding_vector[i1:i2].reshape(1, self._emb_dim)
        return embedding_dict
    
    def initial_vector(self) -> torch.Tensor:
        """
        :return: the vector corresponding to the embeddings given to the constructor.
        """
        return self.pack(self._stored_embeddings)
    
    def number_of_embeddings(self) -> int:
        """
        :return: total number of managed embeddings.
        """
        return len(self._stored_embeddings)
    
    def node_key_to_index(self, key: AgentId) -> int:
        """
        Converts the given AgentId to the ordinal number of its embedding in concatenated vectors.
        :param key: AgentId (must be known during construction).
        :return: the ordinal number of key in concatenated vectors.
        """
        return self._node_key_to_index[key]
    
    def compute_objective(self, embedding_dict: OrderedDict[AgentId, torch.Tensor],
                          nontrivial_diverters: List[AgentId], lambdified_objective: Callable,
                          softmax_temperature: float, probability_smoothing: float) -> torch.Tensor:
        """
        Executes the neural network stored in the RouterGraph on the given embeddings, and then
        computes the given objective as a function of routing probabilities.
        :param embedding_dict: embeddings to be supplied to the neural network.
        :param nontrivial_diverters: list of AgentIds that correspond to nontrivial diverters (i.e., they
            are reachable from the source, and the sink is reachable from both their successors).
        :param lambdified_objective: objective as a function of (the list of) routing probabilities.
            The order of the probabilities must comply with the order of diverters in nontrivial_diverters.
        :param softmax_temperature: temperature (T) hyperparameter.
        :param probability_smoothing: probability smoothing hyperparameter.
        :return: (the value of lambdified_objective, the list of computed probabilities).
        """
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
