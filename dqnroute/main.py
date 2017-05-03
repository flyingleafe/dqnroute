import sys
import signal
import yaml
import networkx as nx

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

    if len(sys.argv) < 2:
        print("Provide path to settings file")
        return

    sfile = open(sys.argv[1])
    run_params = yaml.safe_load(sfile)
    sfile.close()

    G = nx.Graph()
    for e in run_params['network']:
        G.add_edge(**e)

    actorSys = ActorSystem('multiprocQueueBase')
    overlord = actorSys.createActor(Overlord, globalName='overlord')
    actorSys.tell(overlord, OverlordInitMsg(G, run_params['settings']))

    # answer = actorSys.ask(hello, 'hi', 1)
    # print(answer['b'])
    # actorSys.tell(hello, ActorExitRequest())

if __name__ == '__main__':
    main()
