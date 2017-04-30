import sys
import signal
import networkx as nx
import datetime as dt

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

    n = int(next(sys.stdin))
    edges = []
    for i in range(0, n):
        edges.append(parse_edge(next(sys.stdin)))

    package_info = tuple(next(sys.stdin).split())
    n_packages = int(package_info[0])
    pkg_delta = float(package_info[1])

    emulation_settings = tuple(next(sys.stdin).split())
    sync_delta = float(emulation_settings[0])
    period = dt.timedelta(milliseconds=int(emulation_settings[1]))

    G = nx.Graph()
    G.add_weighted_edges_from(edges)

    actorSys = ActorSystem('multiprocQueueBase')
    overlord = actorSys.createActor(Overlord, globalName='overlord')
    actorSys.tell(overlord, OverlordInitMsg(G, (n_packages, pkg_delta), (sync_delta, period)))

    # answer = actorSys.ask(hello, 'hi', 1)
    # print(answer['b'])
    # actorSys.tell(hello, ActorExitRequest())

if __name__ == '__main__':
    main()
