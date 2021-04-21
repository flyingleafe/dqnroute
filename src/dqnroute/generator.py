"""
Data generator for performing a supervised learning procedure
"""
from collections import defaultdict

import numpy as np
import networkx as nx
import pandas as pd
import pprint
import copy
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
    return mk_num_list(tag + '_', n)


def update_network(router, G):
    router.network = G
    router.networkStateChanged()


def _gen_episodes(
        router_type: str,
        one_out: bool,
        factory: RouterFactory,
        num_episodes: int,
        bar=None,
        sinks=None,
        random_seed=None,
        use_full_topology=True
) -> pd.DataFrame:
    print(f'Use full topolology: {use_full_topology}')

    G = factory.topology_graph
    nodes = sorted(G.nodes)
    n = len(nodes)

    amatrix = nx.convert_matrix.to_numpy_array(
        G, nodelist=nodes, weight=factory.edge_weight, dtype=np.float32)
    gstate = np.ravel(amatrix)

    # edges_indexes = []
    # for from_idx, from_node in enumerate(nodes):
    #     for to_idx, to_node in enumerate(nodes):
    #         if amatrix[from_idx][to_idx] > 0:
    #             # edge exists
    #             edges_indexes.append((from_node, to_node))
    #             assert nx.path_weight(G, [from_node, to_node], weight=factory.edge_weight) == amatrix[from_idx][to_idx]
    # edges_indexes = np.array(edges_indexes)

    # best_transitions = defaultdict(dict)
    # lengths = defaultdict(dict)

    # Shortest path preprocessing
    # for start_node in nodes:
    #     for finish_node in nodes:
    #         if start_node != finish_node and nx.has_path(G, start_node, finish_node):
    #             path = nx.dijkstra_path(G, start_node, finish_node, weight=factory.edge_weight)
    #             length = nx.path_weight(G, path, weight=factory.edge_weight)
    #             best_transitions[start_node][finish_node] = path[1] if len(path) > 1 else start_node
    #             lengths[start_node][finish_node] = length

    if sinks is None:
        sinks = nodes

    additional_inputs = None
    routers = {}
    node_dim = 1 if one_out else n

    for rid in nodes:
        router = factory._makeHandler(rid)
        update_network(router, G)
        routers[rid] = router
        if additional_inputs is None:
            additional_inputs = router.additional_inputs

    cols = ['addr', 'dst']

    if 'ppo' in router_type:
        for inp in additional_inputs:
            cols += add_input_cols(inp['tag'], inp.get('dim', n))
        cols += ['next_addr', 'addr_v_func']
    else:
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


    pkg_id = 1
    episode = 0
    while episode < num_episodes:
        current_graph = copy.deepcopy(G)
        # if not use_full_topology:
        #     np.random.shuffle(edges_indexes)
        #     remove_edge_count = np.random.randint(0, len(edges_indexes) // 4)
        #
        #     for edge in edges_indexes[:remove_edge_count]:
        #         u = (edge[0][0], int(edge[0][1]))
        #         v = (edge[1][0], int(edge[1][1]))
        #         current_graph.remove_edge(u, v)

        amatrix = nx.convert_matrix.to_numpy_array(
            current_graph, nodelist=nodes, weight=factory.edge_weight, dtype=np.float32)
        gstate = np.ravel(amatrix)

        dst = random.choice(sinks)
        cur = random.choice(only_reachable(current_graph, dst, nodes))
        if cur == dst:
            continue
        router = routers[cur]
        out_nbrs = current_graph.successors(router.id)
        nbrs = only_reachable(current_graph, dst, out_nbrs)

        if len(nbrs) == 0:
            continue

        episode += 1

        # ppo addition
        if 'ppo' in router_type:
            path = nx.dijkstra_path(G, cur, dst, weight=factory.edge_weight)
            # print(path, cur, dst)
            next_addr = path[1]
            full_path_length = -nx.path_weight(G, path, weight=factory.edge_weight)

            row = [cur[1], dst[1]] + gstate.tolist() + [next_addr[1], full_path_length]
            df.loc[len(df)] = row
        else:
            pkg = Package(pkg_id, DEF_PKG_SIZE, dst, 0, None)
            state = list(router._getNNState(pkg, nbrs))

            def plen_func(v):
                plen = nx.dijkstra_path_length(G, v, dst, weight=factory.edge_weight)
                elen = G.get_edge_data(cur, v)[factory.edge_weight]
                return -(plen + elen)

            if one_out:
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


def gen_episodes(
        router_type: str,
        num_episodes: int,
        context: str,
        one_out=True,
        sinks=None,
        bar=None,
        random_seed=None,
        router_params={},
        save_path=None,
        ignore_saved=False,
        run_params={},
        use_full_topology=True
) -> pd.DataFrame:
    if save_path is not None:
        if not ignore_saved and os.path.isfile(save_path):
            df = pd.read_csv(save_path, index_col=False)
            if bar is not None:
                bar.update(num_episodes)
            return df

    RunnerClass = NetworkRunner if context == 'network' else ConveyorsRunner

    router_params['random_init'] = True
    params_override = {
        'settings': {'router': {router_type: {'random_init': True}}}
    }

    runner = RunnerClass(router_type=router_type, params_override=params_override, run_params=run_params)
    if runner.context == 'network':
        factory = runner.world.factory
    else:
        factory = runner.world.factory.sub_factory

    df = _gen_episodes(
        router_type, one_out, factory, num_episodes, sinks=sinks, bar=bar, random_seed=random_seed,
        use_full_topology=use_full_topology)

    if save_path is not None:
        df.to_csv(save_path, index=False)
    return df
