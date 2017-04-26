import sys
import os
import signal
import yaml
import time
import networkx as nx
import argparse as ap

from thespian.actors import *
from dqnroute.overlord import RoutersOverlord
from dqnroute.messages import RoutersOverlordInitMsg, ReportRequest

stop_it = False

def sigint_handler(signal, frame):
    global stop_it
    actorSys = ActorSystem()
    print("Ctrl-C is hit, reporting results...")
    overlord = actorSys.createActor(RoutersOverlord, globalName='overlord')
    actorSys.ask(overlord, ReportRequest(None))
    print("Shutting down actor system...")
    actorSys.shutdown()
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

    actorSys = ActorSystem('multiprocQueueBase')
    overlord = actorSys.createActor(RoutersOverlord, globalName='overlord')
    actorSys.tell(overlord, RoutersOverlordInitMsg(graph=G,
                                                   settings=run_params['settings'],
                                                   results_file=args.results_file,
                                                   logfile=args.logfile,
                                                   router_type=args.router_type))
    while not stop_it:
        next(sys.stdin)

    # answer = actorSys.ask(hello, 'hi', 1)
    # print(answer['b'])
    # actorSys.tell(hello, ActorExitRequest())

if __name__ == '__main__':
    main()
