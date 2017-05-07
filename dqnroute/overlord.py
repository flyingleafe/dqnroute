import random
import networkx as nx
import datetime as dt

from thespian.actors import *
from more_itertools import peekable

from messages import *
from event_series import EventSeries
from time_actor import Synchronizer, AbstractTimeActor
from router import SimpleQRouter, LinkStateRouter
from utils import gen_network_actions
# from dqn_router import DQNRouter

class Overlord(Actor):
    """Elder actor to start system and rule them all"""

    def __init__(self):
        self.router_counts = {}
        self.times_data = None
        self.cur_time = 0
        self.cur_sum_time = 0
        self.cur_count = 0
        self.results_file = None
        self.log_file = None

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
        results_file = message.results_file
        log_file = message.logfile

        pkg_distr = settings['pkg_distr']
        sync_settings = settings['synchronizer']
        logging_settings = settings['logging']
        router_settings = settings['router']

        if results_file is not None:
            self.results_file = open(results_file, 'w')

        if log_file is not None:
            self.log_file = open(log_file, 'a')
            router_settings['full_log'] = True
        else:
            router_settings['full_log'] = False

        self.times_data = EventSeries(logging_settings['delta'])

        synchronizer = self.createActor(Synchronizer, globalName='synchronizer')
        pkg_sender = self.createActor(PkgSender, globalName='pkg_sender')

        routers = {}
        for n in G:
            routers[n] = self.createActor(LinkStateRouter)

        print("Starting routers")
        for n in G:
            cur_router = routers[n]
            neighbors_addrs = G.neighbors(n)
            self.send(cur_router, LinkStateInitMsg(network_addr=n,
                                                   neighbors={k: G.get_edge_data(n, k) for k in neighbors_addrs},
                                                   network=routers,
                                                   **router_settings))

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
        for (idx, row) in pkg.route.iterrows():
            k = int(row['cur_node'])
            try:
                self.router_counts[k] += 1
            except KeyError:
                self.router_counts[k] = 1

        travel_time = message.time - pkg.start_time
        self.times_data.logEvent(message.time, travel_time)
        print("PACKAGE #{} DONE: path time {}, route len: {}".format(pkg.id, travel_time, len(pkg.route)))
        if self.log_file is not None:
            pkg.route.to_csv(self.log_file, header=False, index=False)

    def reportResults(self):
        for (n, count) in self.router_counts.items():
            print(n, count)
        results_df = self.times_data.getSeries()
        if self.results_file is not None:
            results_df.to_csv(self.results_file)
            self.results_file.close()
        if self.log_file is not None:
            self.log_file.close()

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
                if isinstance(e, IncomingPkgEvent):
                    print("PACKAGE #{} SENT".format(e.getContents().id))
                elif isinstance(e, LinkBreakMsg):
                    print("LINK END {} BROKE".format(e.neighbor))
                elif isinstance(e, LinkRestoreMsg):
                    print("LINK END {} RESTORED".format(e.neighbor))
                self.resendEventDelayed(target, e, self.sync_delta)
        except StopIteration:
            pass

    def _pkgGen(self, network, pkg_distr):
        addrs = list(network.keys())
        for (action, cur_time, params) in gen_network_actions(addrs, pkg_distr):
            if action == 'send_pkg':
                pkg_id, s, d, size = params
                pkg = Package(pkg_id, size, d, cur_time + self.sync_delta, None)
                yield (network[s], IncomingPkgEvent(cur_time, self.myAddress, pkg))
            elif action == 'break_link':
                u, v = params
                yield (network[u], LinkBreakMsg(cur_time, self.myAddress, v))
                yield (network[v], LinkBreakMsg(cur_time, self.myAddress, u))
            elif action == 'restore_link':
                u, v = params
                yield (network[u], LinkRestoreMsg(cur_time, self.myAddress, v))
                yield (network[v], LinkRestoreMsg(cur_time, self.myAddress, u))
            else:
                raise Exception('Unexpected action type: ' + action)
