import sys
import os
import signal
import yaml
import time
import networkx as nx
import argparse as ap
import simpy
import random
import logging

from dqnroute import *

def main():
    logging.basicConfig(stream=sys.stderr, format='%(levelname)s: %(message)s',
                        level=logging.INFO)

    parser = ap.ArgumentParser(description='Routing emulator')
    parser.add_argument('settings_file', metavar='settings_file', type=str,
                        help='Path to run settings file')
    parser.add_argument('results_file', metavar='results_file', type=str,
                        help='Path to results .csv')

    parser.add_argument('router_type', metavar='router_type',
                        choices=['simple_q', 'pred_q', 'link_state', 'dqn'],
                        help='Router type')

    parser.add_argument('--logfile', dest='logfile', default=None,
                        help='Path to routing data')

    args = parser.parse_args()

    router_type = args.router_type

    with open(args.settings_file) as sfile:
        run_params = yaml.safe_load(sfile)

    if args.logfile is not None:
        try:
            os.remove(args.logfile)
        except FileNotFoundError:
            pass

    series = event_series(500, ['count', 'sum', 'min', 'max'])
    series = run_network_scenario(run_params, router_type, series)
    series.getSeries.to_csv(args.logfile)

if __name__ == '__main__':
    main()
