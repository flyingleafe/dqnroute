import networkx as nx
import numpy as np

def mk_current_neural_state(G, time, pkg, node_addr):
    n = len(G.nodes())
    k = node_addr
    d = pkg.dst
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
    for m in G.neighbors(k):
        data[off + k] = 1
        off += n
    for i in range(0, n):
        for j in range(0, n):
            if G.has_edge(i, j):
                data[off + i*n + j] = 1
                off += n*n
    for i in range(off, dlen):
        data[i] = -1000000
    for m in G.neighbors(k):
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

def get_data_cols(n):
    return meta_cols + get_feature_cols(n) + get_amatrix_cols(n) + get_target_cols(n)
