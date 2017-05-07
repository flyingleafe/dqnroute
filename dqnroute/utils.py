import random
import networkx as nx
import numpy as np

def mk_current_neural_state(G, time, pkg, node_addr):
    n = len(G.nodes())
    k = node_addr
    d = pkg.dst
    neighbors = G.neighbors(k)
    dlen = 3 + 4*n + n*n
    data = np.zeros(dlen)
    data[0] = time
    data[1] = pkg.id
    data[2] = k
    off = 3
    data[off + d] = 1
    off += n
    data[off + k] = 1
    off += n
    for m in neighbors:
        data[off + m] = 1
    off += n
    for i in range(0, n):
        for j in range(0, n):
            if G.has_edge(i, j):
                data[off + i*n + j] = 1
    off += n*n
    for i in range(off, dlen):
        data[i] = -1000000
    for m in neighbors:
        data[off + m] = -(nx.dijkstra_path_length(G, m, d) + \
                          G.get_edge_data(k, m)['weight'])
    return data

def mk_num_list(s, n):
    return list(map(lambda k: s+str(k), range(0, n)))

meta_cols = ['time', 'pkg_id', 'cur_node']

def get_target_cols(n):
    return mk_num_list('predict_', 10)

def get_dst_cols(n):
    return mk_num_list('dst_', n)

def get_addr_cols(n):
    return mk_num_list('addr_', n)

def get_neighbors_cols(n):
    return mk_num_list('neighbors_', n)

def get_feature_cols(n):
    return get_dst_cols(n) + get_addr_cols(n) + get_neighbors_cols(n)

def get_amatrix_cols(n):
    res = []
    for m in range(0, n):
        s = 'amatrix_'+str(m)+'_'
        res += mk_num_list(s, n)
    return res

def get_amatrix_triangle_cols(n):
    res = []
    for i in range(0, n):
        for j in range(i+1, n):
            res.append('amatrix_'+str(i)+'_'+str(j))
    return res

def get_data_cols(n):
    return meta_cols + get_feature_cols(n) + get_amatrix_cols(n) + get_target_cols(n)

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
