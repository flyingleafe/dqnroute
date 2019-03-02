import sys
import os
import signal
import yaml
import time
import networkx as nx
import argparse as ap

from thespian.actors import *
from dqnroute.overlord import ConveyorsOverlord
from dqnroute.messages import ConveyorsOverlordInitMsg, ReportRequest

stop_it = False

def sigint_handler(signal, frame):
    global stop_it
    actorSys = ActorSystem()
    print("Ctrl-C is hit, reporting results...")
    overlord = actorSys.createActor(ConveyorsOverlord, globalName='overlord')
    actorSys.ask(overlord, ReportRequest(None))
    print("Shutting down actor system...")
    actorSys.shutdown()
    stop_it = True

def main():
    global stop_it
    signal.signal(signal.SIGINT, sigint_handler)

    parser = ap.ArgumentParser(description='Baggage handling emulator')
    parser.add_argument('settings_file', metavar='settings_file', type=str,
                        help='Path to run settings file')
    parser.add_argument('results_file', metavar='results_file', type=str,
                        help='Path to results .csv')

    parser.add_argument('conveyor_type', metavar='conveyor_type',
                        choices=['simple_q', 'pred_q', 'link_state', 'dqn'],
                        help='Router type')

    parser.add_argument('--logfile', dest='logfile', default=None,
                        help='Path to routing data')

    args = parser.parse_args()

    sfile = open(args.settings_file)
    run_params = yaml.safe_load(sfile)
    sfile.close()

    if args.logfile is not None:
        try:
            os.remove(args.logfile)
        except FileNotFoundError:
            pass

    actorSys = ActorSystem('multiprocQueueBase')
    overlord = actorSys.createActor(ConveyorsOverlord, globalName='overlord')
    actorSys.tell(overlord, ConveyorsOverlordInitMsg(configuration=run_params['configuration'],
                                                     settings=run_params['settings'],
                                                     results_file=args.results_file,
                                                     logfile=args.logfile,
                                                     router_type=args.conveyor_type))
    while not stop_it:
        next(sys.stdin)

if __name__ == '__main__':
    main()
