import yaml
import networkx as nx
import pandas as pd
import numpy as np
import argparse as ap

def main():
    parser = ap.ArgumentParser(description='Path CSV generator (without full emulation)')
    parser.add_argument('settings_file', metavar='settings_file', type=str,
                        help='Path to run settings file')
    parser.add_argument('logfile', metavar='logfile', type=str,
                        help='Path to results .csv')

    args = parser.parse_args()

    sfile = open(args.settings_file)
    run_params = yaml.safe_load(sfile)
    sfile.close()

    G = nx.Graph()
    for e in run_params['network']:
        G.add_edge(e['u'], e['v'], weight=e['latency'])

    logfile = open(args.logfile, 'a')

    

    logfile.close()

if __name__ == '__main__':
    main()
