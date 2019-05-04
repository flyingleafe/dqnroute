import warnings
import networkx as nx
import numpy as np
import scipy.sparse as sp

from typing import Union


class Embedding(object):
    """
    Abstract class for graph node embeddings
    """
    def __init__(self, dim, **kwargs):
        self.dim = dim

    def fit(self, graph: Union[nx.DiGraph, np.ndarray], **kwargs):
        raise NotImplementedError()

    def transform(self, nodes):
        raise NotImplementedError()


class HOPEEmbedding(Embedding):
    def __init__(self, dim, proximity='katz', beta=0.01, **kwargs):
        if dim % 2 != 0:
            dim -= dim % 2
            print('HOPE supports only even embedding dimensions; falling back to {}'.format(dim))

        if proximity not in {'katz', 'common-neighbors'}:
            raise Exception('Unsupported proximity measure: ' + proximity)

        super().__init__(dim, **kwargs)
        self.proximity = proximity
        self.beta = beta
        self._W = None

    def fit(self, graph: Union[nx.DiGraph, np.ndarray], weight='weight'):
        if type(graph) == nx.DiGraph:
            A = nx.to_numpy_matrix(graph, nodelist=sorted(graph.nodes), weight=weight)
            n = graph.number_of_nodes()
        else:
            A = np.mat(graph)
            n = A.shape[0]

        if self.proximity == 'katz':
            M_g = np.eye(n) - self.beta * A
            M_l = self.beta * A
        elif self.proximity == 'common-neighbors':
            M_g = np.eye(n)
            M_l = A * A
        S = np.dot(np.linalg.inv(M_g), M_l)

        u, s, vt = lg.svds(S, k=self.dim // 2)
        X1 = np.dot(u, np.diag(np.sqrt(s)))
        X2 = np.dot(vt.T, np.diag(np.sqrt(s)))
        self._W = np.concatenate((X1, X2), axis=1)

    def transform(self, idx):
        return self._W[idx]


class LaplacianEigenmap(Embedding):
    def __init__(self, dim, renormalize_weights=True, weight_transform='heat',
                 temp=1.0, **kwargs):
        super().__init__(dim, **kwargs)
        self.renormalize_weights = renormalize_weights
        self.weight_transform = weight_transform
        self.temp = temp
        self._X = None

    def fit(self, graph: Union[nx.DiGraph, np.ndarray], weight='weight'):
        if type(graph) == np.ndarray:
            graph = nx.from_numpy_array(graph, create_using=nx.DiGraph)
            weight = 'weight'

        graph = graph.to_undirected()

        if weight is not None:
            if self.renormalize_weights:
                sum_w = sum([ps[weight] for _, _, ps in graph.edges(data=True)])
                avg_w = sum_w / len(graph.edges())
                for u, v, ps in graph.edges(data=True):
                    graph[u][v][weight] /= avg_w

            if self.weight_transform == 'inv':
                for u, v, ps in graph.edges(data=True):
                    graph[u][v][weight] = 1 / ps[weight]

            elif self.weight_transform == 'heat':
                for u, v, ps in graph.edges(data=True):
                    w = ps[weight]
                    graph[u][v][weight] = np.exp(-w*w)

        A = nx.to_scipy_sparse_matrix(graph, nodelist=sorted(graph.nodes),
                                      weight=weight, format='csr', dtype=np.float32)
        n, m = A.shape
        diags = A.sum(axis=1)
        D = sp.spdiags(diags.flatten(), [0], m, n, format='csr')
        L = D - A

        values, vectors = sp.linalg.eigsh(L, k=self.dim + 1, M=D, which='SM')
        self._X = vectors[:, 1:]

        if weight is not None and self.renormalize_weights:
            self._X *= avg_w

    def transform(self, idx):
        return self._X[idx]


_emb_classes = {
    'hope': HOPEEmbedding,
    'lap': LaplacianEigenmap,
}

def get_embedding(alg: str, **kwargs):
    try:
        return _emb_classes[alg](**kwargs)
    except KeyError:
        raise Exception('Unsupported embedding algorithm: ' + alg)

