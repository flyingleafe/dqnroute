import sys
import networkx as nx

from thespian.actors import *
from overlord import Overlord
from messages import OverlordInitMsg

def main():
    n = int(next(sys.stdin))
    edges = []
    for i in range(0, n):
        edges.append(tuple([int(x) for x in next(sys.stdin).split()]))

    package_info = tuple(next(sys.stdin).split())
    n_packages = int(package_info[0])
    delta = float(package_info[1])

    G = nx.Graph()
    G.add_edges_from(edges)

    actorSys = ActorSystem('multiprocQueueBase')
    overlord = actorSys.createActor(Overlord, globalName='overlord')
    actorSys.tell(overlord, OverlordInitMsg(G, (n_packages, delta)))

    # answer = actorSys.ask(hello, 'hi', 1)
    # print(answer['b'])
    # actorSys.tell(hello, ActorExitRequest())

if __name__ == '__main__':
    main()
