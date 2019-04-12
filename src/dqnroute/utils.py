import random
import networkx as nx
import numpy as np
import torch
import itertools as it

from typing import NewType

from .constants import INFTY

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

def make_network_graph(edge_list) -> nx.DiGraph:
    """
    Creates a computer network graph (with symmetric edges)
    from edge list
    """

    def read_edge(e):
        new_e = e.copy()
        u = new_e.pop('u')
        v = new_e.pop('v')
        return (u, v, new_e)

    DG = nx.DiGraph()
    for e in edge_list:
        u, v, params = read_edge(e)
        DG.add_edge(u, v, **params)
        DG.add_edge(v, u, **params)
    return DG

def make_conveyor_graph(layout) -> nx.DiGraph:
    """
    Creates a conveyor network graph from conveyor system layout.
    """

    DG = nx.DiGraph()
    for conveyor in layout:
        for sec_id, section in conveyor.items():
            length = section['length']
            try:
                upn = section['upstream_neighbor']
                DG.add_edge(sec_id, upn, length=length)
            except KeyError:
                pass

            try:
                ajn = section['adjacent_neighbor']
                DG.add_edge(sec_id, ajn, length=length)
            except KeyError:
                pass
    return DG

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

def sec_neighbors_list(section_info):
    res = []
    ups = section_info.get('upstream_neighbor', None)
    if ups is not None:
        res.append(ups)
    adj = section_info.get('adjacent_neighbor', None)
    if adj is not None:
        res.append(adj)
    return res

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
