import random
import networkx as nx
from thespian.actors import *
from more_itertools import peekable

from messages import *
from time_actor import Synchronizer, AbstractTimeActor
from router import SimpleQRouter

class Overlord(Actor):
    """Elder actor to start system and rule them all"""

    def receiveMessage(self, message, sender):
        if isinstance(message, OverlordInitMsg):
            self.startSystem(message)
        else:
            pass

    def startSystem(self, message):
        G = message.graph
        (n_packages, pack_delta) = message.packet_distr
        (sync_delta, period) = message.emulation_settings

        synchronizer = self.createActor(Synchronizer, globalName='synchronizer')
        pkg_sender = self.createActor(PkgSender, globalName='pkg_sender')

        routers = {}
        for n in G:
            routers[n] = self.createActor(SimpleQRouter)

        for n in G:
            cur_router = routers[n]
            self.send(cur_router, RouterInitMsg(n, G.neighbors(n), routers))

        self.send(pkg_sender, PkgSenderInitMsg(n_packages, pack_delta, sync_delta, routers))
        self.senf(synchronizer, SynchronizerInitMsg(list(routers.values()) + [pkg_sender], sync_delta, period))

class PkgSender(AbstractTimeActor):
    def __init__(self):
        self.pkg_iterator = None
        self.sync_delta = None

    def initialize(self, message, sender):
        self.sync_delta = message.sync_delta
        self.pkg_iterator = peekable(self._pkgGen(message.network, message.n_packages, message.pkg_delta))

    def handleTick(self, time):
        try:
            while self.pkg_iterator.peek()[1].time <= time:
                (target, e) = self.pkg_iterator.next()
                self.resendEventDelayed(target, e, self.sync_delta)
        except StopIteration:
            pass

    def _pkgGen(self, network, n_packages, pkg_delta):
        addrs = list(network.keys())
        cur_time = 0
        for i in range(0, n_packages):
            [s, d] = random.sample(addrs, 2)
            pkg = Package(d, cur_time + self.sync_delta, None)
            yield (network[d], PackageMsg(cur_time, self.myAddress, pkg))
            cur_time += pkg_delta
