import random
import networkx as nx
import datetime as dt

from thespian.actors import *
from more_itertools import peekable

from messages import *
from event_series import EventSeries
from time_actor import Synchronizer, AbstractTimeActor
from router import SimpleQRouter

class Overlord(Actor):
    """Elder actor to start system and rule them all"""

    def __init__(self):
        self.router_counts = {}
        self.avg_times = EventSeries(10)
        self.avg_route_lens = EventSeries(10)
        self.cur_time = 0
        self.cur_sum_time = 0
        self.cur_count = 0

    def receiveMessage(self, message, sender):
        if isinstance(message, OverlordInitMsg):
            self.startSystem(message)
        elif isinstance(message, PkgDoneMsg):
            self.recordPkg(message)
        elif isinstance(message, ReportRequest):
            self.reportResults()
            self.send(sender, ReportDone(None))
        else:
            pass

    def startSystem(self, message):
        print("Overlord is started")

        G = message.graph
        settings = message.settings
        pkg_distr = settings['pkg_distr']
        sync_settings = settings['synchronizer']

        synchronizer = self.createActor(Synchronizer, globalName='synchronizer')
        pkg_sender = self.createActor(PkgSender, globalName='pkg_sender')

        routers = {}
        for n in G:
            routers[n] = self.createActor(SimpleQRouter)


        print("Starting routers")
        for n in G:
            cur_router = routers[n]
            neighbors_addrs = G.neighbors(n)
            self.send(cur_router, SimpleQRouterInitMsg(network_addr=n,
                                                       neighbors={k: G.get_edge_data(n, k) for k in neighbors_addrs},
                                                       network=routers,
                                                       learning_rate=0.2))

        print("Starting pkg sender")
        self.send(pkg_sender, PkgSenderInitMsg(pkg_distr,
                                               sync_settings['delta'],
                                               routers))

        print("Starting synchronizer")
        self.send(synchronizer, SynchronizerInitMsg(list(routers.values()) + [pkg_sender],
                                                    sync_settings['delta'],
                                                    dt.timedelta(milliseconds=sync_settings['period'])))

    def recordPkg(self, message):
        pkg = message.getContents()
        for k in pkg.route:
            try:
                self.router_counts[k] += 1
            except KeyError:
                self.router_counts[k] = 1

        travel_time = message.time - pkg.start_time
        self.avg_times.logEvent(message.time, travel_time)
        self.avg_route_lens.logEvent(message.time, len(pkg.route))
        print("PACKAGE #{} DONE: path time {}, route: {}".format(pkg.id, travel_time, pkg.route))

    def reportResults(self):
        for (n, count) in self.router_counts.items():
            print(n, count)
        print(self.avg_times.getSeries())
        print(self.avg_route_lens.getSeries())

class PkgSender(AbstractTimeActor):
    """Sends series of packages according to given settings"""

    def __init__(self):
        self.pkg_iterator = None
        self.sync_delta = None

    def initialize(self, message, sender):
        self.sync_delta = message.sync_delta
        self.pkg_iterator = peekable(self._pkgGen(message.network, message.pkg_distr))

    def handleTick(self, time):
        try:
            while self.pkg_iterator.peek()[1].time <= time:
                (target, e) = self.pkg_iterator.next()
                print("PACKAGE #{} SENT".format(e.getContents().id))
                self.resendEventDelayed(target, e, self.sync_delta)
        except StopIteration:
            pass

    def _pkgGen(self, network, distr_list):
        addrs = list(network.keys())
        cur_time = 0
        pkg_id = 1
        for distr in distr_list:
            n_packages = distr['pkg_number']
            pkg_delta = distr['delta']
            for i in range(0, n_packages):
                [s, d] = random.sample(addrs, 2)
                pkg = Package(pkg_id, 1024, d, cur_time + self.sync_delta, None)
                yield (network[s], IncomingPkgEvent(cur_time, self.myAddress, pkg))
                cur_time += pkg_delta
                pkg_id += 1
