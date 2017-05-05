import sys
import os
import signal
import yaml
import networkx as nx
import argparse as ap

from thespian.actors import *
from overlord import Overlord
from messages import OverlordInitMsg, ReportRequest

def sigint_handler(signal, frame):
    actorSys = ActorSystem()
    print("Ctrl-C is hit, reporting results...")
    overlord = actorSys.createActor(Overlord, globalName='overlord')
    actorSys.ask(overlord, ReportRequest(None))
    print("Shutting down actor system...")
    actorSys.shutdown()

def parse_edge(s):
    [a, b, w] = s.split()
    return (int(a), int(b), float(w))

def main():
    signal.signal(signal.SIGINT, sigint_handler)

    parser = ap.ArgumentParser(description='Routing emulator')
    parser.add_argument('settings_file', metavar='settings_file', type=str,
                        help='Path to run settings file')
    parser.add_argument('results_file', metavar='results_file', type=str,
                        help='Path to results .csv')

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
        os.remove(args.logfile)

    actorSys = ActorSystem('multiprocQueueBase')
    overlord = actorSys.createActor(Overlord, globalName='overlord')
    actorSys.tell(overlord, OverlordInitMsg(G, run_params['settings'], args.results_file, args.logfile))

    # answer = actorSys.ask(hello, 'hi', 1)
    # print(answer['b'])
    # actorSys.tell(hello, ActorExitRequest())

if __name__ == '__main__':
    main()
