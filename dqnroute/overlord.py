import random
import time
import networkx as nx
import datetime as dt

from thespian.actors import *
from more_itertools import peekable

from .messages import *
from .event_series import EventSeries
from .time_actor import Synchronizer, AbstractTimeActor
from .router import SimpleQRouter, PredictiveQRouter, LinkStateRouter
from .conveyor import LinkStateConveyor, SimpleQConveyor
from .utils import *
from .dqn_router import DQNRouter
from .dqn_conveyor import DQNConveyor

class Overlord(Actor):
    """Elder actor to start system and rule them all"""

    def __init__(self):
        self.root = None
        self.router_counts = {}
        self.times_data = None
        self.results_file = None
        self.log_file = None
        self.router_type = None
        self.routers = {}
        self.answered_inits = {}
        self.settings = None
        self.router_sequential_init = False
        self.init_msgs = {}
        self.pkgs_received = 0
        self.total_pkgs_needed = 0

    def receiveMessage(self, message, sender):
        if isinstance(message, OverlordInitMsg):
            self.startSystem(message, sender)
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
                            time.sleep(0.2)
                        self.init_msgs = {}
                    self.finishSystemStart()
                elif self.router_sequential_init:
                    nxt = message.child_id + 1
                    self.send(self.routers[nxt], self.init_msgs[nxt])
        else:
            pass

    def startSystem(self, message, sender):
        print("Overlord is started")
        self.root = sender
        results_file = message.results_file
        log_file = message.logfile

        if results_file is not None:
            self.results_file = open(results_file, 'w')
        if log_file is not None:
            self.log_file = open(log_file, 'a')

    def finishSystemStart(self):
        pass

    def recordPkg(self, message):
        pass

    def reportResults(self):
        pass

    def sendInitMsgs(self):
        if self.router_sequential_init:
            self.send(self.routers[0], self.init_msgs[0])
        else:
            for (n, target) in self.routers.items():
                self.send(target, self.init_msgs[n])
            self.init_msgs = {}
        print("Waiting for routers to initialize...")

    def startSynchronizer(self, nodes):
        sync_settings = self.settings['synchronizer']
        synchronizer = self.createActor(Synchronizer, globalName='synchronizer')
        print("Starting synchronizer")
        period = sync_settings['period']
        if type(period) == dict:
            period = period[self.router_type]
        self.send(synchronizer, SynchronizerInitMsg(nodes, sync_settings['delta'], period))

class RoutersOverlord(Overlord):
    def startSystem(self, message, sender):
        super().startSystem(message, sender)

        if not isinstance(message, RoutersOverlordInitMsg):
            raise Exception("Invalid init msg for RoutersOverlord!")

        G = message.graph
        settings = message.settings
        self.router_type = message.router_type
        self.total_pkgs_needed = sum(map(lambda d: d.get('pkg_number', 0), settings['pkg_distr']['sequence']))

        self.settings = settings
        logging_settings = settings['logging']
        router_settings = settings['router']
        router_settings['full_log'] = self.log_file is not None

        self.times_data = EventSeries(logging_settings['delta'])

        router_class = None
        router_init_msg_class = None
        if self.router_type == 'link_state':
            print('Using link-state router algorithm')
            router_class = LinkStateRouter
            router_init_msg_class = LinkStateInitMsg
            self.router_sequential_init = True
        elif self.router_type == 'simple_q':
            print('Using Simple Q-routing router algorithm')
            router_class = SimpleQRouter
            router_init_msg_class = SimpleQRouterInitMsg
        elif self.router_type == 'pred_q':
            print('Using Predictive Q-routing router algorithm')
            router_class = PredictiveQRouter
            router_init_msg_class = PredictiveQRouterInitMsg
        elif self.router_type == 'dqn':
            print('Using DQN router algorithm')
            router_class = DQNRouter
            router_init_msg_class = DQNRouterInitMsg
            self.router_sequential_init = True
        else:
            raise Exception('Unknown router type: ' + self.router_type)

        self.routers = {}
        for n in G:
            self.routers[n] = self.createActor(router_class)

        print("Starting routers")
        routers_individ_settings = settings.get('routers_individual', {})
        for n in G:
            cur_router = self.routers[n]
            neighbors_addrs = G.neighbors(n)
            cur_settings = router_settings
            cur_settings.update(routers_individ_settings.get(n, {}))
            msg = router_init_msg_class(network_addr=n,
                                        neighbors={k: G.get_edge_data(n, k) for k in neighbors_addrs},
                                        network=self.routers,
                                        **cur_settings)
            self.init_msgs[n] = msg

        self.sendInitMsgs()

    def finishSystemStart(self):
        pkg_distr = self.settings['pkg_distr']
        sync_settings = self.settings['synchronizer']
        state_size = self.settings['router'].get('rnn_size', 64)

        pkg_sender = self.createActor(PkgSender, globalName='pkg_sender')
        print("Starting pkg sender")
        self.send(pkg_sender, PkgSenderInitMsg(pkg_distr,
                                               sync_settings['delta'],
                                               self.routers,
                                               state_size))

        nodes = list(self.routers.values()) + [pkg_sender]
        self.startSynchronizer(nodes)

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

        self.pkgs_received += 1
        if self.pkgs_received >= self.total_pkgs_needed:
            time.sleep(2)
            print("#!#!#! shutdown")

    def reportResults(self):
        for (n, count) in self.router_counts.items():
            print(n, count)
        results_df = self.times_data.getSeries()
        if self.results_file is not None:
            results_df.to_csv(self.results_file)
            self.results_file.close()
        if self.log_file is not None:
            self.log_file.close()

class ConveyorsOverlord(Overlord):
    def __init__(self):
        super().__init__()
        self.sec_network = {}
        self.energy_data = None

    def startSystem(self, message, sender):
        super().startSystem(message, sender)

        if not isinstance(message, ConveyorsOverlordInitMsg):
            raise Exception("Invalid init msg for ConveyorsOverlord!")

        configuration = message.configuration
        settings = message.settings
        self.router_type = message.router_type
        self.total_pkgs_needed = sum(map(lambda d: d.get('bags_number', 0), settings['bags_distr']['sequence']))

        self.settings = settings
        logging_settings = settings['logging']
        conveyor_settings = settings['router']
        conveyor_settings['full_log'] = self.log_file is not None

        conveyor_class = None
        conveyor_init_msg_class = None
        self.router_sequential_init = True
        if self.router_type == 'link_state':
            print('Using link-state router algorithm')
            conveyor_class = LinkStateConveyor
            conveyor_init_msg_class = LinkStateConveyorInitMsg
        elif self.router_type == 'simple_q':
            print('Using Simple Q-routing router algorithm')
            conveyor_class = SimpleQConveyor
            conveyor_init_msg_class = SimpleQConveyorInitMsg
        # elif self.router_type == 'pred_q':
            # print('Using Predictive Q-routing router algorithm')
            # router_class = PredictiveQRouter
            # router_init_msg_class = PredictiveQRouterInitMsg
        elif self.router_type == 'dqn':
            print('Using DQN router algorithm')
            conveyor_class = DQNConveyor
            conveyor_init_msg_class = DQNConveyorInitMsg
        else:
            raise Exception('Unknown router type: ' + self.router_type)

        self.times_data = EventSeries(logging_settings['delta'], prefix='time')
        self.energy_data = EventSeries(logging_settings['delta'], prefix='energy', func='sum')

        self.routers = {}
        self.sec_network = {}
        all_sections = {}
        for (n, cfg) in enumerate(configuration):
            self.routers[n] = self.createActor(conveyor_class)
            for (sec, sec_info) in cfg['sections'].items():
                all_sections[sec] = sec_info
                self.sec_network[sec] = self.routers[n]

        print("Starting conveyors")
        for (n, cfg) in enumerate(configuration):
            conv_settings = conveyor_settings.copy()
            conv_settings.update(cfg.get('settings', {}))
            msg = conveyor_init_msg_class(network_addr=n,
                                          network=self.sec_network,
                                          sections=cfg['sections'],
                                          all_sections=all_sections,
                                          **conv_settings)
            self.init_msgs[n] = msg

        self.sendInitMsgs()

    def finishSystemStart(self):
        bags_distr = self.settings['bags_distr']
        sync_settings = self.settings['synchronizer']
        state_size = self.settings['router'].get('rnn_size', 64)

        bag_sender = self.createActor(BagSender, globalName='bag_sender')
        print("Starting bag sender")
        self.send(bag_sender, BagSenderInitMsg(pkg_distr=bags_distr,
                                               sync_delta=sync_settings['delta'],
                                               network=self.sec_network,
                                               state_size=state_size,
                                               sources=self.settings['router']['sources'],
                                               sinks=self.settings['router']['sinks']))

        nodes = list(self.routers.values()) + [bag_sender]
        self.startSynchronizer(nodes)

    def recordPkg(self, message):
        bag = message.getContents()
        for (idx, row) in bag.route.iterrows():
            k = int(row['cur_node'])
            try:
                self.router_counts[k] += 1
            except KeyError:
                self.router_counts[k] = 1

        travel_time = message.time - bag.start_time
        self.times_data.logEvent(message.time, travel_time)
        energy_spent = bag.energy_spent
        self.energy_data.logEvent(message.time, energy_spent)

        print("BAG #{} DONE: path time {}, energy overhead {}, route: {}"
              .format(bag.id, travel_time, energy_spent, list(bag.route['cur_node'].astype(int))))

        if self.log_file is not None:
            pkg.route.to_csv(self.log_file, header=False, index=False)

        self.pkgs_received += 1
        if self.pkgs_received >= self.total_pkgs_needed:
            time.sleep(2)
            print("#!#!#! shutdown")

    def reportResults(self):
        for (n, count) in self.router_counts.items():
            print(n, count)
        times_df = self.times_data.getSeries()
        energy_df = self.energy_data.getSeries()
        results_df = times_df.merge(energy_df, on='time')

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
        self.state_size = 64

    def initialize(self, message, sender):
        self.sync_delta = message.sync_delta
        self.state_size = message.state_size
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
                pkg = Package(pkg_id, size, d, cur_time + self.sync_delta, self.state_size, None)
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


class BagSender(AbstractTimeActor):
    """Sends series of bags according to given settings"""

    def __init__(self):
        self.bag_iterator = None
        self.sync_delta = None
        self.state_size = 64

    def initialize(self, message, sender):
        self.sync_delta = message.sync_delta
        self.state_size = message.state_size
        self.pkg_iterator = peekable(self._pkgGen(message.network,
                                                  message.sources,
                                                  message.sinks,
                                                  message.pkg_distr))

    def handleTick(self, time):
        try:
            while self.pkg_iterator.peek()[1].time <= time:
                (target, e) = self.pkg_iterator.next()
                if isinstance(e, IncomingLuggageEvent):
                    print("BAG #{} SENT".format(e.getContents().id))
                # elif isinstance(e, LinkBreakMsg):
                    # print("LINK END {} BROKE".format(e.neighbor))
                # elif isinstance(e, LinkRestoreMsg):
                    # print("LINK END {} RESTORED".format(e.neighbor))
                self.resendEventDelayed(target, e, self.sync_delta)
        except StopIteration:
            pass

    def _pkgGen(self, network, sources, sinks, bag_distr):
        for (action, cur_time, params) in gen_conveyor_actions(sources, sinks, bag_distr):
            if action == 'send_bag':
                bag_id, s, d = params
                ftime = cur_time + self.sync_delta
                bag = Bag(bag_id, d, ftime, ftime, self.state_size, None)
                yield (network[s], IncomingLuggageEvent(cur_time, self.myAddress, s, -1, bag))
            elif action == 'break_sections':
                for sec in params:
                    pass
            else:
                raise Exception('Unexpected action type: ' + action)

