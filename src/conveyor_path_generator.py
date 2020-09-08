import yaml
import os
import random
import networkx as nx
import pandas as pd
import numpy as np
import argparse as ap

from dqnroute.utils import *
from dqnroute.messages import Bag

def mk_random_bin_vector(n, allowed_positions):
    v = np.zeros(n)
    for p in allowed_positions:
        v[p] = random.choice((0, 1))
    return v

def mk_work_status_vector(n, nbs, conv_sections):
    v = mk_random_bin_vector(n, nbs)
    for sec in conv_sections:
        v[sec] = 1
    return v

def main():
    parser = ap.ArgumentParser(description='Conveyor path CSV generator (without full emulation)')
    parser.add_argument('settings_file', metavar='settings_file', type=str,
                        help='Path to run settings file')
    parser.add_argument('logfile', metavar='logfile', type=str,
                        help='Path to results .csv')

    args = parser.parse_args()

    sfile = open(args.settings_file)
    run_params = yaml.safe_load(sfile)
    sfile.close()

    G = nx.DiGraph()
    links_data = {}
    configuration = run_params['configuration']
    settings = run_params['settings']
    conveyor_common_cfg = settings['router']
    speed = conveyor_common_cfg['speed']
    all_sections = {}
    sec_conveyors = {}

    for (i, belt) in enumerate(configuration):
        all_sections.update(belt['sections'])
        for sec in belt['sections'].keys():
            sec_conveyors[sec] = i

    for (sec_num, sec_info) in all_sections.items():
        G.add_node(sec_num)
        for n in sec_neighbors_list(sec_info):
            weight = all_sections[n]['length'] / speed
            G.add_edge(sec_num, n, weight=weight)

    try:
        os.remove(args.logfile)
    except FileNotFoundError:
        pass

    logfile = open(args.logfile, 'a')

    df = pd.DataFrame(columns=['belt_id']+get_conveyor_data_cols(len(G.nodes())))
    s_delta = settings['synchronizer']['delta']

    for (action, cur_time, params) in gen_conveyor_actions(conveyor_common_cfg['sources'],
                                                           conveyor_common_cfg['sinks'],
                                                           settings['bags_distr']):
        if action == 'send_bag':
            bag_id, s, d = params
            path = nx.dijkstra_path(G, s, d)
            time = cur_time + s_delta
            bag = Bag(bag_id, d, time, time, 64, None)
            for (i, n) in enumerate(path):
                if i < len(path) - 1:
                    cur_conveyor = sec_conveyors[n]
                    conv_sections = list(configuration[cur_conveyor]['sections'].keys())
                    working_nb_vec = mk_work_status_vector(len(all_sections), G.neighbors(n), conv_sections)
                    _row = mk_current_neural_state(G, time, bag, n, working_nb_vec)
                    row = np.concatenate(([cur_conveyor], _row))
                    df.loc[len(df)] = row
                    time += G.get_edge_data(n, path[i+1])['weight']
            df.to_csv(logfile, header=False, index=False)
            df.drop(df.index, inplace=True)
            print("bag #{} done".format(bag_id))
        # elif action == 'break_link':
        #     u, v = params
        #     G.remove_edge(u, v)
        #     print('removed link ({}, {})'.format(u, v))
        # elif action == 'restore_link':
        #     u, v = params
        #     w = links_data[(u, v)]['latency']
        #     G.add_edge(u, v, weight=w)
        #     print('restored link ({}, {})'.format(u, v))
        else:
            raise Exception('Unexpected action type: ' + action)

    logfile.close()

if __name__ == '__main__':
    main()
