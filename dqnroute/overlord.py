import random
import networkx as nx
import datetime as dt

from thespian.actors import *
from more_itertools import peekable

from messages import *
from event_series import EventSeries
from time_actor import Synchronizer, AbstractTimeActor
from router import SimpleQRouter, PredictiveQRouter, LinkStateRouter
from utils import gen_network_actions
from dqn_router import DQNRouter

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
        self.routers = {}
        self.answered_inits = {}
        self.settings = None
        self.G = None
        self.router_sequential_init = False
        self.init_msgs = {}

    def receiveMessage(self, message, sender):
        if isinstance(message, OverlordInitMsg):
            self.startSystem(message)
        elif isinstance(message, PkgDoneMsg):
            self.recordPkg(message)
        elif isinstance(message, ReportRequest):
            self.reportResults()
            self.send(sender, ReportDone(None))
        elif isinstance(message, FinishInitMsg):
            if message.child_id is not None:
                self.answered_inits[message.child_id] = True
                print("Router {} initialized".format(message.child_id))
                if len(self.answered_inits) == len(self.routers):
                    if self.router_sequential_init:
                        for (n, target) in self.routers.items():
                            self.send(target, RouterFinalizeInitMsg())
                        self.init_msgs = {}
                    self.finishSystemStart()
                elif self.router_sequential_init:
                    nxt = message.child_id + 1
                    self.send(self.routers[nxt], self.init_msgs[nxt])
        else:
            pass

    def startSystem(self, message):
        print("Overlord is started")

        G = message.graph
        self.G = G
        settings = message.settings
        results_file = message.results_file
        log_file = message.logfile
        router_type = message.router_type

        self.settings = settings
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

        router_class = None
        router_init_msg_class = None
        if router_type == 'link_state':
            print('Using link-state router algorithm')
            router_class = LinkStateRouter
            router_init_msg_class = LinkStateInitMsg
        elif router_type == 'simple_q':
            print('Using Simple Q-routing router algorithm')
            router_class = SimpleQRouter
            router_init_msg_class = SimpleQRouterInitMsg
        elif router_type == 'pred_q':
            print('Using Predictive Q-routing router algorithm')
            router_class = PredictiveQRouter
            router_init_msg_class = PredictiveQRouterInitMsg
        elif router_type == 'dqn':
            print('Using DQN router algorithm')
            router_class = DQNRouter
            router_init_msg_class = DQNRouterInitMsg
            self.router_sequential_init = True
        else:
            raise Exception('Unknown router type: ' + router_type)

        self.routers = {}
        for n in G:
            self.routers[n] = self.createActor(router_class)

        print("Starting routers")
        for n in G:
            cur_router = self.routers[n]
            neighbors_addrs = G.neighbors(n)
            msg = router_init_msg_class(network_addr=n,
                                        neighbors={k: G.get_edge_data(n, k) for k in neighbors_addrs},
                                        network=self.routers,
                                        **router_settings)
            self.init_msgs[n] = msg

        if self.router_sequential_init:
            self.send(self.routers[0], self.init_msgs[0])
        else:
            for (n, target) in self.routers.items():
                self.send(target, self.init_msgs[n])
            self.init_msgs = {}

        print("Waiting for routers to initialize...")

    def finishSystemStart(self):
        pkg_distr = self.settings['pkg_distr']
        sync_settings = self.settings['synchronizer']

        synchronizer = self.createActor(Synchronizer, globalName='synchronizer')
        pkg_sender = self.createActor(PkgSender, globalName='pkg_sender')

        print("Starting pkg sender")
        self.send(pkg_sender, PkgSenderInitMsg(pkg_distr,
                                               sync_settings['delta'],
                                               self.routers))

        print("Starting synchronizer")
        self.send(synchronizer, SynchronizerInitMsg(list(self.routers.values()) + [pkg_sender],
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
        print("PACKAGE #{} DONE: path time {}, route: {}".format(pkg.id, travel_time, list(pkg.route['cur_node'].astype(int))))
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
