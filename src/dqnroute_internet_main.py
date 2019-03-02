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

from dqnroute.simpy_router import SimPyDumbRouter
from dqnroute.messages import Package

TIMEOUT_SEND_RANDOM_PKG = 10
stop_it = False
logger = logging.getLogger("dqnroute_internet_main")

def sendRandomPkg(env, routerList):
    while True:
        yield env.timeout(TIMEOUT_SEND_RANDOM_PKG)
        srcIndex = random.randint(0, len(routerList) - 1)
        dstIndex = random.randint(0, len(routerList) - 1)
        src = routerList[srcIndex]
        dstID = routerList[dstIndex].id
        pkg = Package(0, 0, dstID, 0, 0, None) # create empty packet

        print("Sending random pkg {} -> {} at time {}".format(src.id, dstID, env.now))
        src.sendPackage(pkg)
        logger.debug("Sending random pkg continue time: %d", env.now)

def sigint_handler(signal, frame):
    global stop_it
    print("Ctrl-C is hit, reporting results...")
    print("Shutting down actor system...")
    stop_it = True

def read_edge(e):
    new_e = e.copy()
    new_e['u_of_edge'] = new_e.pop('u')
    new_e['v_of_edge'] = new_e.pop('v')
    return new_e

def main():
    global stop_it
    signal.signal(signal.SIGINT, sigint_handler)
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

    sfile = open(args.settings_file)
    run_params = yaml.safe_load(sfile)
    sfile.close()

    G = nx.Graph()
    for e in run_params['network']:
        G.add_edge(**read_edge(e))

    if args.logfile is not None:
        try:
            os.remove(args.logfile)
        except FileNotFoundError:
            pass

    env = simpy.Environment()
    routers = {}
    for node in G.nodes():
        routers[node] = SimPyDumbRouter(env, node)
    for node in G.nodes():
        out_routers = [routers[v] for (_, v) in G.edges(node)]
        routers[node].setNeighbours(out_routers)
    env.process(sendRandomPkg(env, routers))
    env.run(until=40)

if __name__ == '__main__':
    main()
