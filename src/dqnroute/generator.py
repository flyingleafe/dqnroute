"""
Data generator for performing a supervised learning procedure
"""
import numpy as np
import networkx as nx
import pandas as pd
import pprint
import os

from .utils import *
from .constants import *
from .agents import *
from .simulation import *

def add_input_cols(tag, dim):
    if tag == 'amatrix':
        return get_amatrix_cols(dim)
    else:
        return mk_num_list(tag + '_', dim)

def unsqueeze(arr, min_d=2):
    if len(arr.shape) == 0:
        arr = np.array([arr])
    if len(arr.shape) < min_d:
        return arr.reshape(arr.shape[0], -1)
    return arr

def _cols(tag, n):
    if n == 1:
        return [tag]
    return mk_num_list(tag+'_', n)

def update_network(router, G):
    router.network = G
    router.networkStateChanged()

def _gen_episodes(factory: RouterFactory, num_episodes: int, bar=None,
                  sinks=None, random_seed=None) -> pd.DataFrame:
    RouterClass = factory.RouterClass
    if not issubclass(RouterClass, DQNRouter):
        raise Exception('Trying to generate pre-training dataset not for DQN-* router')

    G = factory.topology_graph
    nodes = sorted(G.nodes)
    n = len(nodes)
    if sinks is None:
        sinks = nodes

    additional_inputs = None
    routers = {}
    node_dim = 1 if issubclass(RouterClass, DQNRouterOO) else n

    for rid in nodes:
        router = factory._makeHandler(rid)
        update_network(router, G)
        routers[rid] = router
        if additional_inputs is None:
            additional_inputs = router.additional_inputs

    cols = ['addr', 'dst']
    if node_dim == 1:
        cols.append('neighbour')
    else:
        cols += get_neighbors_cols(node_dim)

    for inp in additional_inputs:
        cols += add_input_cols(inp['tag'], inp.get('dim', n))

    if node_dim == 1:
        cols.append('predict')
    else:
        cols += get_target_cols(n)

    df = pd.DataFrame(columns=cols)

    if random_seed is not None:
        set_random_seed(random_seed)

    pkg_id = 1
    episode = 0
    while episode < num_episodes:
        dst = random.choice(sinks)
        cur = random.choice(only_reachable(G, dst, nodes))
        router = routers[cur]
        out_nbrs = G.successors(router.id)
        nbrs = only_reachable(G, dst, out_nbrs)

        if len(nbrs) == 0:
            continue
        episode += 1

        pkg = Package(pkg_id, DEF_PKG_SIZE, dst, 0, None)
        state = list(router._getNNState(pkg, nbrs))

        def plen_func(v):
            plen = nx.dijkstra_path_length(G, v, dst, weight=factory.edge_weight)
            elen = G.get_edge_data(cur, v)[factory.edge_weight]
            return -(plen + elen)

        if issubclass(RouterClass, DQNRouterOO):
            predict = np.fromiter(map(plen_func, nbrs), dtype=np.float32)
            state.append(predict)
            cat_state = np.concatenate([unsqueeze(y) for y in state], axis=1)
            for row in cat_state:
                df.loc[len(df)] = row
        else:
            predict = np.fromiter(map(lambda i: plen_func(('router', i)) if ('router', i) in nbrs else -INFTY,
                                      range(n)),
                                  dtype=np.float32)
            state.append(predict)
            state_ = [unsqueeze(y, 1) for y in state]
            # pprint.pprint(state_)
            cat_state = np.concatenate(state_)
            df.loc[len(df)] = cat_state

        if bar is not None:
            bar.update(1)

    return df

def gen_episodes(num_episodes: int, context: str, one_out=True, sinks=None, bar=None,
                 random_seed = None, router_params={}, save_path=None,
                 ignore_saved=False, **kwargs):
    if save_path is not None:
        if not ignore_saved and os.path.isfile(save_path):
            df = pd.read_csv(save_path, index_col=False)
            if bar is not None:
                bar.update(num_episodes)
            return df

    RunnerClass = NetworkRunner if context == 'network' else ConveyorsRunner

    if one_out:
        router_type = 'dqn_oneout'
    else:
        router_type = 'dqn'

    router_params['random_init'] = True
    params_override = {
        'settings': {'router': {router_type: {'random_init': True}}}
    }

    runner = RunnerClass(router_type=router_type, params_override=params_override, **kwargs)
    if runner.context == 'network':
        factory = runner.world.factory
    else:
        factory = runner.world.factory.sub_factory
    df = _gen_episodes(factory, num_episodes, sinks=sinks, bar=bar, random_seed=random_seed)

    if save_path is not None:
        df.to_csv(save_path, index=False)
    return df
