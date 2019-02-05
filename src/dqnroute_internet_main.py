import sys
import os
import signal
import yaml
import time
import networkx as nx
import argparse as ap
import simpy
import random

from dqnroute.simpy_router import SimPyDumbRouter
from dqnroute.messages import Package

TIMEOUT_SEND_RANDOM_PKG = 10
stop_it = False

def sendRandomPkg(env, routerList):
    while True:
        yield env.timeout(TIMEOUT_SEND_RANDOM_PKG)
        srcIndex = random.randint(0, len(routerList) - 1)
        dstIndex = random.randint(0, len(routerList) - 1)
        src = routerList[srcIndex]
        dstID = routerList[dstIndex].id
        pkg = Package(0, 0, dstID, 0, 0, None) # create empty packet

        #print("Sending random pkg from router {} -> {} at {}".format(src.id, dstID, env.now))
        src.sendPackage(pkg)
        #print("Sending random pkg continue time: ", env.now)
        
def sigint_handler(signal, frame):
    global stop_it
    print("Ctrl-C is hit, reporting results...")
    print("Shutting down actor system...")
    stop_it = True

def main():
    global stop_it
    signal.signal(signal.SIGINT, sigint_handler)

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
        G.add_edge(**e)

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
    
    #while not stop_it:
    #    next(sys.stdin)

    # answer = actorSys.ask(hello, 'hi', 1)
    # print(answer['b'])
    # actorSys.tell(hello, ActorExitRequest())

if __name__ == '__main__':
    main()
