"""
Data generator for performing a supervised learning procedure
"""
import numpy as np
import networkx as nx
import pandas as pd

from .utils import *
from .constants import *
from .agents import *

def add_input_cols(tag, dim):
    if tag == 'amatrix':
        return get_amatrix_cols(dim)
    else:
        return mk_num_list(tag + '_', dim)

def reassign_id(router: DQNRouter, new_id: int):
    router.id = new_id
    router.out_neighbours = set([v for (_, v) in router.network.out_edges(router.id)])
    router.in_neighbours = set([v for (v, _) in router.network.in_edges(router.id)])
    return router

def unsqueeze(arr):
    if len(arr.shape) < 2:
        return arr.reshape(len(arr), -1)
    return arr

def gen_network_episodes(G: nx.DiGraph, num_episodes: int, RouterClass,
                         sinks = None, random_seed = None, additional_inputs=[], **kwargs) -> pd.DataFrame:

    if not issubclass(RouterClass, DQNRouter):
        raise Exception('Trying to generate pre-training dataset not for DQN-* router')

    nodes = sorted(G.nodes)
    n = len(nodes)
    if sinks is None:
        sinks = nodes

    cols = ['addr', 'dst']
    if issubclass(RouterClass, DQNRouterOO):
        cols.append('neighbour')
    else:
        cols += get_neighbors_cols(n)

    for inp in additional_inputs:
        cols += add_input_cols(inp['tag'], inp.get('dim', n))

    if issubclass(RouterClass, DQNRouterOO):
        cols.append('predict')
    else:
        cols += get_target_cols(n)

    df = pd.DataFrame(columns=cols)
    router = RouterClass(env=DynamicEnv(), router_id=0, adj_links=G.adj[0],
                         additional_inputs=additional_inputs, nodes=nodes,
                         out_neighbours=[v for (_, v) in G.out_edges(0)],
                         in_neighbours=[v for (v, _) in G.in_edges(0)],
                         layers=[64, 64], activation='relu',
                         batch_size=1, mem_capacity=1, **kwargs)

    router.network = G

    if random_seed is not None:
        set_random_seed(random_seed)

    pkg_id = 1
    for i in range(num_episodes):
        dst = random.choice(sinks)
        cur = random.choice(only_reachable(G, dst, nodes))
        router = reassign_id(router, cur)
        nbrs = only_reachable(G, dst, router.out_neighbours)

        pkg = Package(pkg_id, DEF_PKG_SIZE, dst, 0, None)
        state = router._getNNState(pkg, nbrs=nbrs)
        nbr_state = state[2]

        plen_func = lambda v: -(nx.dijkstra_path_length(G, v, dst) + G.get_edge_data(cur, v)['weight'])
        if issubclass(RouterClass, DQNRouterOO):
            predict = np.fromiter(map(plen_func, nbr_state), dtype=np.float32)
            state.append(predict)
            cat_state = np.concatenate([unsqueeze(y) for y in state], axis=1)
            for row in cat_state:
                df.loc[len(df)] = row
        else:
            predict = np.fromiter(map(lambda i, up: plen_func(i) if up else -INFTY, range(n)),
                                  dtype=np.float32)
            state.append(predict)
            cat_state = np.concatenate(state)
            df.loc[len(df)] = cat_state

    return df
