import networkx as nx
import numpy as np

from gem.embedding.hope import HOPE

class Embedding(object):
    """
    Abstract class for graph node embeddings
    """
    def __init__(self, dim, **kwargs):
        self.dim = dim

    def fit(self, graph: nx.DiGraph, **kwargs):
        raise NotImplementedError()

    def get_embedding(self, node):
        raise NotImplementedError()

class HOPEEmbedding(Embedding):
    def __init__(self, dim, measure='katz', beta=0.01, **kwargs):
        if measure != 'katz':
            raise NotImplementedError('Only Katz index is currently supported :(')

        if dim % 2 != 0:
            dim -= dim % 2
            print('HOPE supports only even embedding dimensions; falling back to {}'.format(dim))

        super().__init__(dim, **kwargs)
        self._impl = HOPE(d=dim, beta=beta)
        self._W = None

    def fit(self, graph: nx.DiGraph, weight='weight'):
        self._W = self._impl.learn_embedding(graph=graph, weight=weight)[0]

    def get_embedding(self, idx):
        return self._W[idx]
