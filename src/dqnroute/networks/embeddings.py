import networkx as nx
import numpy as np

from typing import Union
from gem.embedding.hope import HOPE

class Embedding(object):
    """
    Abstract class for graph node embeddings
    """
    def __init__(self, dim, **kwargs):
        self.dim = dim

    def fit(self, graph: Union[nx.DiGraph, np.ndarray], **kwargs):
        raise NotImplementedError()

    def get_embedding(self, node):
        raise NotImplementedError()

class HOPEEmbedding(Embedding):
    def __init__(self, dim, proximity='katz', beta=0.01, **kwargs):
        if dim % 2 != 0:
            dim -= dim % 2
            print('HOPE supports only even embedding dimensions; falling back to {}'.format(dim))

        super().__init__(dim, **kwargs)
        self._impl = HOPE(d=dim, proximity=proximity, beta=beta)
        self._W = None

    def fit(self, graph: Union[nx.DiGraph, np.ndarray], weight='weight'):
        if type(graph) == nx.DiGraph:
            self._W = self._impl.learn_embedding(graph=graph, weight=weight)[0]
        else:
            self._W = self._impl.learn_embedding(amatrix=graph)[0]

    def get_embedding(self, idx):
        return self._W[idx]
