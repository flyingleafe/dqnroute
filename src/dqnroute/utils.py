import random
import struct
import hashlib
import base64
import networkx as nx
import numpy as np
import torch
import itertools as it

from typing import NewType, Tuple, TypeVar, Union, List, Callable, Optional

from .constants import INFTY, DEF_PKG_SIZE

##
# Some ubuquitous type aliases
#

AgentId = Tuple[str, int]
InterfaceId = int

##
# Misc
#

def set_random_seed(seed: int):
    """
    Sets given random seed in all relevant RNGs
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def memoize(func):
    mp = {}
    def memfoo(x):
        try:
            return mp[x]
        except KeyError:
            r = func(x)
            mp[x] = r
            return r
    return memfoo

def empty_gen():
    yield from ()

##
# Graph construction and config processing
#

def make_network_graph(edge_list) -> nx.Graph:
    """
    Creates a computer network graph from edge list
    """

    def read_edge(e):
        new_e = e.copy()
        u = new_e.pop('u')
        v = new_e.pop('v')
        return (u, v, new_e)

    G = nx.Graph()
    for e in edge_list:
        u, v, params = read_edge(e)
        G.add_edge(('router', u), ('router', v), **params)
    return G

def gen_network_graph(gen) -> nx.Graph:
    """
    Generates a random computer network graph given a
    generator parameters
    """
    gen_type = gen['type']

    if gen_type == 'barabasi-albert':
        G = nx.barabasi_albert_graph(gen['n'], gen['m'], seed=gen['seed'])
        np.random.seed(gen['seed'])
        for u, v in G.edges():
            G[u][v]['bandwidth'] = DEF_PKG_SIZE
            G[u][v]['latency'] = np.random.randint(gen['min-latency'], gen['max-latency'])
        return G
    else:
        raise Exception('Unsupported graph generator type: {}'.format(gen_type))

def agent_type(aid):
    if type(aid) == tuple:
        return aid[0]
    return aid

def agent_idx(aid):
    if type(aid) == tuple:
        return aid[1]
    return aid

def make_conveyor_topology_graph(config) -> nx.DiGraph:
    """
    Creates a conveyor network graph from conveyor system layout.
    """
    sources = config['sources']
    conveyors = config['conveyors']
    diverters = config['diverters']

    conv_sections = {conv_id: [] for conv_id in conveyors.keys()}

    for (src_id, src) in sources.items():
        conv = src['upstream_conv']
        conv_sections[conv].append(('source', src_id, 0))

    for (dv_id, dv) in diverters.items():
        conv = dv['conveyor']
        pos = dv['pos']
        up_conv = dv['upstream_conv']
        conv_sections[conv].append(('diverter', dv_id, pos))
        conv_sections[up_conv].append(('diverter', dv_id, 0))

    junction_idx = 0
    for (conv_id, conv) in conveyors.items():
        l = conv['length']
        up = conv['upstream']

        if up['type'] == 'sink':
            conv_sections[conv_id].append(('sink', up['idx'], l))
        elif up['type'] == 'conveyor':
            up_id = up['idx']
            up_pos = up['pos']
            conv_sections[conv_id].append(('junction', junction_idx, l))
            conv_sections[up_id].append(('junction', junction_idx, up_pos))
            junction_idx += 1
        else:
            raise Exception('Invalid conveyor upstream type: ' + up['type'])

    DG = nx.DiGraph()
    for (conv_id, sections) in conv_sections.items():
        sections.sort(key=lambda s: s[2])
        assert sections[0][2] == 0, \
            "No node at the beginning of conveyor!"
        assert sections[-1][2] == conveyors[conv_id]['length'], \
            "No node at the end of conveyor!"

        for i in range(1, len(sections)):
            u = sections[i-1][:-1]
            v = sections[i][:-1]
            u_pos = sections[i-1][-1]
            v_pos = sections[i][-1]
            edge_len = v_pos - u_pos

            assert edge_len >= 2, "Conveyor section is way too short!"
            DG.add_edge(u, v, length=edge_len, conveyor=conv_id)

            if (i > 1) or (u[0] != 'diverter'):
                DG.nodes[u]['conveyor'] = conv_id
                DG.nodes[u]['conveyor_pos'] = u_pos
            if (i < len(sections) - 1) or (v[0] != 'junction'):
                DG.nodes[v]['conveyor'] = conv_id
                DG.nodes[v]['conveyor_pos'] = v_pos

    return DG

def make_conveyor_conn_graph(config) -> nx.Graph:
    """
    Creates a connection graph between controllers in conveyor system.
    """
    G = nx.Graph()
    sinks = config['sinks']
    sources = config['sources']
    conveyors = config['conveyors']
    diverters = config['diverters']

    for sink in sinks:
        G.add_node(('sink', sink))
    for src in sources.keys():
        G.add_node(('source', src))
    for conv in conveyors.keys():
        G.add_node(('conveyor', conv))
    for dv in diverters.keys():
        G.add_node(('diverter', dv))

    for src_id, src in sources.items():
        G.add_edge(('source', src_id), ('conveyor', src['upstream_conv']))
    for conv_id, conv in conveyors.items():
        up = conv['upstream']
        G.add_edge(('conveyor', conv_id), (up['type'], up['idx']))
    for dv_id, dv in diverters.items():
        G.add_edge(('diverter', dv_id), ('conveyor', dv['conveyor']))
        G.add_edge(('diverter', dv_id), ('conveyor', dv['upstream_conv']))

    return G

def interface_idx(conn_graph: nx.Graph, from_agent: AgentId, to_agent: AgentId) -> int:
    return list(conn_graph.edges(from_agent)).index((from_agent, to_agent))

def resolve_interface(conn_graph: nx.Graph, from_agent: AgentId, int_id: int) -> Tuple[AgentId, int]:
    """
    Given a connection graph, node ID and adjacent edge index, returns
    a node on the other side of that edge and that edge's index
    among other node's edges.
    Edge index -1 is a loopback.
    """
    if int_id == -1:
        to_agent = from_agent
        to_interface = -1
    else:
        to_agent = list(conn_graph.edges(from_agent))[int_id][1]
        to_interface = interface_idx(conn_graph, to_agent, from_agent)
    return to_agent, to_interface

def make_router_cfg(G, router_id):
    """
    Helper which makes valid config for the router controller of a given class
    """
    router_cfg = {
        'nodes': sorted(list(G.nodes())),
        'edges_num': len(G.edges()) # small hack to make link-state initialization simpler
    }
    return router_cfg

def only_reachable(G, v, nodes, inv_paths=True):
    filter_func = lambda u: nx.has_path(G, u, v) if inv_paths else nx.has_path(G, v, u)
    return list(filter(filter_func, nodes))

##
# Generation of training data and manipulations with it
#

def mk_current_neural_state(G, time, pkg, node_addr, *add_data):
    n = len(G.nodes())
    k = node_addr
    d = pkg.dst
    neighbors = []
    if isinstance(G, nx.DiGraph):
        for m in G.neighbors(k):
            if nx.has_path(G, m, d):
                neighbors.append(m)
    else:
        neighbors = G.neighbors(k)

    add_data_len = sum(map(len, add_data))
    dlen = 4 + 2*n + add_data_len + n*n
    data = np.zeros(dlen)
    data[0] = d
    data[1] = k
    data[2] = time
    data[3] = pkg.id
    off = 4
    for m in neighbors:
        data[off + m] = 1
    off += n

    for vec in add_data:
        vl = len(vec)
        data[off:off+vl] = vec
        off += vl

    for i in range(0, n):
        for j in range(0, n):
            if G.has_edge(i, j):
                data[off + i*n + j] = 1
    off += n*n
    for i in range(off, dlen):
        data[i] = -INFTY
    for m in neighbors:
        try:
            data[off + m] = -(nx.dijkstra_path_length(G, m, d) + \
                              G.get_edge_data(k, m)['weight'])
        except nx.exception.NetworkXNoPath:
            data[off + m] = -INFTY
    return data

def dict_min(dct):
    return min(dct.items(), key=lambda x:x[1])

def mk_num_list(s, n):
    return list(map(lambda k: s+str(k), range(0, n)))

meta_cols = ['time', 'pkg_id']
base_cols = ['dst', 'addr']
common_cols = base_cols + meta_cols

@memoize
def get_target_cols(n):
    return mk_num_list('predict_', n)

@memoize
def get_dst_cols(n):
    return mk_num_list('dst_', n)

@memoize
def get_addr_cols(n):
    return mk_num_list('addr_', n)

@memoize
def get_neighbors_cols(n):
    return mk_num_list('neighbors_', n)

@memoize
def get_work_status_cols(n):
    return mk_num_list('work_status_', n)

@memoize
def get_feature_cols(n):
    return get_dst_cols(n) + get_addr_cols(n) + get_neighbors_cols(n)

@memoize
def get_amatrix_cols(n):
    res = []
    for m in range(0, n):
        s = 'amatrix_'+str(m)+'_'
        res += mk_num_list(s, n)
    return res

@memoize
def get_amatrix_triangle_cols(n):
    res = []
    for i in range(0, n):
        for j in range(i+1, n):
            res.append('amatrix_'+str(i)+'_'+str(j))
    return res

@memoize
def get_data_cols(n):
    return common_cols + get_neighbors_cols(n) + get_amatrix_cols(n) + get_target_cols(n)

@memoize
def get_conveyor_data_cols(n):
    return common_cols + get_neighbors_cols(n) + get_work_status_cols(n) + get_amatrix_cols(n) + get_target_cols(n)

def make_batches(size, batch_size):
    num_batches = int(np.ceil(size / float(batch_size)))
    for i in range(0, num_batches):
        yield (i * batch_size, min(size, (i + 1) * batch_size))

def gen_network_actions(addrs, pkg_distr):
    cur_time = 0
    pkg_id = 1
    distr_list = pkg_distr['sequence']
    random.seed(pkg_distr.get('seed', None))
    for distr in distr_list:
        action = distr.get('action', 'send_pkgs')
        if action == 'send_pkgs':
            n_packages = distr['pkg_number']
            pkg_delta = distr['delta']
            sources = distr.get('sources', addrs)
            dests = distr.get('dests', addrs)
            swap = distr.get('swap', 0)
            for i in range(0, n_packages):
                s, d = 0, 0
                while s == d:
                    s = random.choice(sources)
                    d = random.choice(dests)
                if random.random() < swap:
                    d, s = s, d
                yield ('send_pkg', cur_time, (pkg_id, s, d, 1024))
                cur_time += pkg_delta
                pkg_id += 1
        elif action == 'break_link' or action == 'restore_link':
            pause = distr['pause']
            u = distr['u']
            v = distr['v']
            yield (action, cur_time, (u, v))
            cur_time += pause
        else:
            raise Exception('Unexpected action: ' + action)

def gen_conveyor_actions(sources, sinks, bags_distr):
    cur_time = 0
    bag_id = 1
    distr_list = bags_distr['sequence']
    random.seed(bags_distr.get('seed', None))
    for distr in distr_list:
        action = distr.get('action', 'send_bags')
        if action == 'send_bags':
            n_bags = distr['bags_number']
            bag_delta = distr['delta']
            sources_ = distr.get('sources', sources)
            sinks_ = distr.get('sinks', sinks)
            for i in range(0, n_bags):
                s = random.choice(sources_)
                d = random.choice(sinks_)
                yield ('send_bag', cur_time, (bag_id, s, d))
                cur_time += bag_delta
                bag_id += 1
        elif action == 'break_sections' or action == 'restore_sections':
            pause = distr['pause']
            yield (action, cur_time, distr['sections'])
            cur_time += pause
        else:
            raise Exception('Unexpected action: ' + action)

def unique_everseen(iterable, key=None):
    "List unique elements, preserving order. Remember all elements ever seen."
    # unique_everseen('AAAABBBCCDAABBB') --> A B C D
    # unique_everseen('ABBCcAD', str.lower) --> A B C D
    seen = set()
    seen_add = seen.add
    if key is None:
        for element in it.ifilterfalse(seen.__contains__, iterable):
            seen_add(element)
            yield element
    else:
        for element in iterable:
            k = key(element)
            if k not in seen:
                seen_add(k)
                yield element

def transpose_arr(arr):
    sh = arr.shape
    if len(sh) == 1:
        sh = (sh[0], 1)
    a, b = sh
    if a is None:
        a = 1
    if b is None:
        b = 1
    return arr.reshape((b, a))

def reverse_input(inp):
    return list(map(transpose_arr, inp))

def stack_batch(batch):
    if type(batch[0]) == dict:
        return stack_batch_dict(batch)
    else:
        return stack_batch_list(batch)

def stack_batch_dict(batch):
    ss = {}
    for k in batch[0].keys():
        ss[k] = np.vstack([b[k] for b in batch])
        if ss[k].shape[1] == 1:
            ss[k] = ss[k].flatten()
    return ss

def stack_batch_list(batch):
    n = len(batch[0])
    ss = [None]*n
    for i in range(n):
        ss[i] = np.vstack([b[i] for b in batch])
        if ss[i].shape[1] == 1:
            ss[i] = ss[i].flatten()
    return ss

#
# Attribute accessor
#

class DynamicEnv(object):
    """
    Dynamic env is an object which stores a bunch of read-only attributes,
    which might be functions
    """

    def __init__(self, **attrs):
        self._attrs = attrs

    def __getattr__(self, name):
        try:
            return super().__getattribute__('_attrs')[name]
        except KeyError:
            raise AttributeError(name)

    def register(self, name, val):
        self._attrs[name] = val

#
# Stochastic policy distribution
#

Distribution = NewType('Distribution', np.ndarray)

def delta(i: int, n: int) -> Distribution:
    if i >= n:
        raise Exception('Action index is out of bounds')
    d = np.zeros(n)
    d[i] = 1
    return Distribution(d)

def uni(n) -> Distribution:
    return Distribution(np.full(n, 1.0/n))

def softmax(x, t=1.0) -> Distribution:
    ax = np.array(x) / t
    ax -= np.amax(ax)
    e = np.exp(ax)
    sum = np.sum(e)
    if sum == 0:
        return uni(len(ax))
    return Distribution(e / np.sum(e, axis=0))

def sample_distr(distr: Distribution) -> int:
    return np.random.choice(np.arange(len(distr)), p=distr)

def soft_argmax(arr, t=1.0) -> int:
    return sample_distr(softmax(arr, t=t))

#
# Simple datatypes serialization and hashing
#

def to_bytes(data):
    if isinstance(data, bytes):
        return data
    elif isinstance(data, str):
        return data.encode()
    elif isinstance(data, int):
        return struct.pack("!i", data)
    elif isinstance(data, float):
        return struct.pack("!f", data)
    elif isinstance(data, bool):
        return struct.pack("!?", data)

    elif isinstance(data, (set, tuple, list)):
        return b''.join([to_bytes(x) for x in data])

    elif isinstance(data, dict):
        return b''.join([to_bytes((k, v)) for k, v in sorted(data.items())])

    else:
        raise Exception('Unsupported type to serialize: {}'.format(type(data)))

def data_digest(data):
    ba = to_bytes(data)
    m = hashlib.sha256()
    m.update(ba)
    return base64.b16encode(m.digest()).decode('utf-8')

##
# Algorithmic helpers
#

T = TypeVar('T')
X = TypeVar('X')

def binary_search(ls: List[T], diff_func: Callable[[T], X],
                  return_index : bool = False) -> Union[Tuple[T, int], T]:
    """
    Binary search via predicate
    """
    l = 0
    r = len(ls)
    while l < r:
        m = l + (r - l) // 2
        cmp_res = diff_func(ls[m])
        if cmp_res == 0:
            return (ls[m], m) if return_index else ls[m]
        elif cmp_res < 0:
            r = m
        else:
            l = m + 1

    if l > 0 and (abs(diff_func(ls[l-1])) < abs(diff_func(ls[l]))):
        l -= 1
    return (ls[l], l) if return_index else ls[l]

def differs_from(x: T, using = None) -> Callable[[T], X]:
    def _diff(y):
        if using is not None:
            y = using(y)
        return x - y
    return _diff
