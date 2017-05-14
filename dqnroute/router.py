import networkx as nx
import numpy as np
import threading

from thespian.actors import *

from router_mixins import RLAgent, LinkStateHolder
from messages import *
from time_actor import *
from utils import mk_current_neural_state, get_data_cols

class RouterNotInitialized(Exception):
    """
    This is raised then some message which is not `RouterInitMsg` comes first to router
    """

class Router(TimeActor):
    def __init__(self):
        super().__init__()
        self.overlord = None
        self.addr = None
        self.network = {}
        self.network_inv = {}
        self.neighbors = {}
        self.pkg_process_delay = 0
        self.queue_time = 0
        self.queue_count = 0
        self.link_states = {}
        self.full_log = False

    def initialize(self, message, sender):
        super().initialize(message, sender)
        if isinstance(message, RouterInitMsg):
            self.overlord = sender
            self.addr = message.network_addr
            self.neighbors = message.neighbors
            self.network = message.network
            self.pkg_process_delay = message.pkg_process_delay
            self.network_inv = {str(target) : addr for addr, target in message.network.items()}
            self.full_log = message.full_log
            for n in self.neighbors.keys():
                self.link_states[n] = {'transfer_time': 0, 'alive': True}
            return self.addr

    def processEvent(self, event):
        if not self.isInitialized():
            raise RouterNotInitialized("Router has not been initialized!")

        if isinstance(event, IncomingPkgEvent):
            self._enqueuePkg(event.sender, event.getContents())
        elif isinstance(event, ProcessPkgEvent):
            self.queue_count -= 1
            pkg = event.getContents()
            if self.full_log:
                pkg.route_add(self._currentStateData(pkg), self._currentStateCols())
            else:
                pkg.route_add([self.current_time, self.addr], ['time', 'cur_node'])
            print("ROUTER #{} ROUTES PACKAGE {} TO {}".format(self.addr, pkg.id, pkg.dst))
            if pkg.dst == self.addr:
                self.receivePackage(event)
                self.reportPkgDone(pkg, self.current_time)
            else:
                best_neighbor = self.routePackage(pkg)
                is_alive = self.link_states[best_neighbor]['alive']
                if is_alive:
                    self.receivePackage(event)
                    target = self.network[best_neighbor]
                    link_latency = self.neighbors[best_neighbor]['latency']
                    link_bandwidth = self.neighbors[best_neighbor]['bandwidth']
                    transfer_start_time = max(self.current_time, self.link_states[best_neighbor]['transfer_time'])
                    transfer_end_time = transfer_start_time + (pkg.size / link_bandwidth)
                    finish_time = transfer_end_time + link_latency
                    self.link_states[best_neighbor]['transfer_time'] = transfer_end_time
                    self.sendEvent(target, IncomingPkgEvent(finish_time, self.myAddress, pkg))
                else:
                    self.sendToBrokenLink(event.sender, pkg)
        elif isinstance(event, LinkBreakMsg):
            n = event.neighbor
            self.link_states[n]['alive'] = False
            self.breakLink(n)
        elif isinstance(event, LinkRestoreMsg):
            n = event.neighbor
            self.link_states[n]['alive'] = True
            self.restoreLink(n)
        else:
            pass

    def _enqueuePkg(self, sender, pkg):
        self.queue_time = max(self.current_time, self.queue_time) + self.pkg_process_delay
        new_event = ProcessPkgEvent(self.queue_time, sender, pkg)
        self.event_queue.push(new_event)
        self.queue_count += 1

    def isInitialized(self):
        return self.overlord is not None

    def reportPkgDone(self, pkg, time):
        self.send(self.overlord, PkgDoneMsg(time, self.myAddress, pkg))

    def receivePackage(self, pkg_event):
        pass

    def routePackage(self, pkg_event):
        pass

    def sendToBrokenLink(self, sender, pkg):
        self._enqueuePkg(sender, pkg)

    def breakLink(self, v):
        pass

    def restoreLink(self, v):
        pass

    def _nodesList(self):
        return sorted(list(self.network.keys()))

    def _currentStateData(self, pkg):
        pass

    def _currentStateCols(self):
        pass

class LinkStateRouter(Router, LinkStateHolder):
    def __init__(self):
        LinkStateHolder.__init__(self)
        super().__init__()
        self.outgoing_pkgs_num = {}

    def initialize(self, message, sender):
        my_id = super().initialize(message, sender)
        if isinstance(message, LinkStateInitMsg):
            self.initGraph(self.addr, self.network, self.neighbors, self.link_states)
            self._announceLinkState()
            self._cur_state_cols = get_data_cols(len(self.network))
            for n in self.neighbors.keys():
                self.outgoing_pkgs_num[n] = 0
        return my_id

    def _announceLinkState(self):
        announcement = self.mkLSAnnouncement(self.addr)
        self._broadcastAnnouncement(announcement, self.myAddress)
        self.seq_num += 1

    def _broadcastAnnouncement(self, announcement, sender):
        for n in self.neighbors.keys():
            if sender != self.network[n]:
                self.sendServiceMsg(self.network[n], announcement)

    def receiveServiceMsg(self, message, sender):
        super().receiveServiceMsg(message, sender)
        if isinstance(message, LinkStateAnnouncement):
            if self.processLSAnnouncement(message, self.network.keys()):
                self._broadcastAnnouncement(message, sender)
        elif isinstance(message, PkgReturnedMsg):
            n = self.network_inv[str(sender)]
            if self.outgoing_pkgs_num[n] > 0:
                self.outgoing_pkgs_num[n] -= 1

    def breakLink(self, v):
        self.lsBreakLink(self.addr, v)
        self._announceLinkState()

    def restoreLink(self, v):
        self.lsRestoreLink(self.addr, v)
        self._announceLinkState()

    def isInitialized(self):
        return super().isInitialized() and (len(self.announcements) == len(self.network))

    def receivePackage(self, pkg_event):
        self.sendServiceMsg(pkg_event.sender, PkgReturnedMsg(None))

    def routePackage(self, pkg):
        d = pkg.dst
        path = nx.dijkstra_path(self.network_graph, self.addr, d)
        return path[1]

    def _currentStateData(self, pkg):
        return mk_current_neural_state(self.network_graph, self.outgoing_pkgs_num, self.current_time, pkg, self.addr)

    def _currentStateCols(self):
        return self._cur_state_cols

class QRouter(Router, RLAgent):
    def __init__(self):
        super().__init__()
        self.reward_pending = {}

    def receivePackage(self, pkg_event):
        pkg = pkg_event.getContents()
        self.sendServiceMsg(pkg_event.sender, self.mkRewardMsg(pkg))

    def routePackage(self, pkg):
        state = self.getState(pkg)
        self.reward_pending[pkg.id] = state
        return self.act(state)

    def receiveServiceMsg(self, message, sender):
        super().receiveServiceMsg(message, sender)
        if isinstance(message, RewardMsg):
            prev_state = self.reward_pending[message.pkg_id]
            del self.reward_pending[message.pkg_id]
            self.observe(self.mkSample(message, prev_state, sender))

    def mkRewardMsg(self, pkg):
        pass

    def getState(self, pkg):
        pass

    def mkSample(self, message, prev_state, sender):
        pass

MAX_EPSILON = 1
MIN_EPSILON = 0.01
LAMBDA = 0.001

class SimpleQRouter(QRouter):
    """Original Q-routing algorithm"""

    def __init__(self):
        super().__init__()
        self.Q = {}
        self.U = {}
        self.learning_rate = None
        self.broken_links = {}
        self.broken_link_Qs = {}
        self.steps = 0

    def initialize(self, message, sender):
        my_id = super().initialize(message, sender)
        if isinstance(message, SimpleQRouterInitMsg):
            self.learning_rate = message.learning_rate
            for n in self.network.keys():
                self.Q[n] = {}
                self.U[n] = {}
                for (k, data) in self.neighbors.items():
                    if k == n:
                        self.Q[n][k] = 0
                    else:
                        self.Q[n][k] = 10
                    self.U[n][k] = 0
        return my_id

    def breakLink(self, v):
        self.broken_links[v] = True
        # broken_Qs = {}
        # for n in self.network.keys():
        #     broken_Qs[n] = self.Q[n][v]
        #     self.Q[n][v] = 100500
        # self.broken_link_Qs[v] = broken_Qs

    def restoreLink(self, v):
        del self.broken_links[v]
        # broken_Qs = self.broken_link_Qs[v]
        # for (n, val) in broken_Qs.items():
        #     self.Q[n][v] = val
        # del self.broken_link_Qs[v]

    def _mkBestEstimate(self, d):
        result_q = {}
        for n in self.neighbors.keys():
            if n not in self.broken_links:
                result_q[n] = self.Q[d][n]
        return dict_min(result_q)

    def mkRewardMsg(self, pkg):
        d = pkg.dst
        best_estimate = 0 if self.addr == d else self._mkBestEstimate(d)[1]
        return SimpleRewardMsg(pkg.id, self.current_time, best_estimate, d)

    def mkSample(self, message, prev_state, sender):
        if isinstance(message, SimpleRewardMsg):
            sender_addr = self.network_inv[str(sender)]
            sent_time = prev_state[0]
            new_estimate = message.estimate + (message.cur_time - sent_time)
            return (message.dst, sender_addr, new_estimate)
        else:
            raise Exception("Unsupported type of reward msg!")

    def getState(self, pkg):
        return (self.current_time, pkg.dst)

    def act(self, state):
        d = state[1]
        return self._mkBestEstimate(d)[0]

    def observe(self, sample):
        (dst, sender_addr, new_estimate) = sample
        delta = self.learning_rate * (new_estimate - self.Q[dst][sender_addr])
        # period = self.current_time - self.U[dst][sender_addr]
        # if period != 0:
        self.Q[dst][sender_addr] += delta
            # self.U[dst][sender_addr] = self.current_time

    def sendToBrokenLink(self, sender, pkg):
        super().sendToBrokenLink(sender, pkg)

    def _currentStateData(self, pkg):
        return [self.current_time, pkg.id]

    def _currentStateCols(self):
        return ['time', 'pkg_id']


class PredictiveQRouter(QRouter):
    """Predictive Q-routing"""

    def __init__(self):
        super().__init__()
        self.Q = {}
        self.B = {}
        self.R = {}
        self.U = {}
        self.learning_rate = None
        self.broken_links = {}
        self.steps = 0

    def initialize(self, message, sender):
        my_id = super().initialize(message, sender)
        if isinstance(message, PredictiveQRouterInitMsg):
            self.learning_rate = message.learning_rate
            self.beta_rate = message.beta_rate
            self.gamma_rate = message.gamma_rate
            for n in self.network.keys():
                self.Q[n] = {}
                self.B[n] = {}
                self.R[n] = {}
                self.U[n] = {}
                for (k, data) in self.neighbors.items():
                    if k == n:
                        self.Q[n][k] = 0
                        self.B[n][k] = 0
                    else:
                        self.Q[n][k] = 5
                        self.B[n][k] = 5
                    self.U[n][k] = 0
                    self.R[n][k] = 0
        return my_id

    def breakLink(self, v):
        self.broken_links[v] = True

    def restoreLink(self, v):
        del self.broken_links[v]

    def _mkBestEstimate(self, d):
        result_q = {}
        for y in self.neighbors.keys():
            if y not in self.broken_links:
                dt = self.current_time - self.U[d][y]
                result_q[y] = max(self.Q[d][y] + dt*self.R[d][y], self.B[d][y])

        return dict_min(result_q)

    def mkRewardMsg(self, pkg):
        d = pkg.dst
        best_estimate = 0 if self.addr == d else dict_min(self.Q[d])[1]
        return PredictiveRewardMsg(pkg.id, self.current_time, best_estimate, d)

    def mkSample(self, message, prev_state, sender):
        if isinstance(message, PredictiveRewardMsg):
            sender_addr = self.network_inv[str(sender)]
            sent_time = prev_state[0]
            new_estimate = message.estimate + (message.cur_time - sent_time)
            return (message.dst, sender_addr, new_estimate)
        else:
            raise Exception("Unsupported type of reward msg!")

    def getState(self, pkg):
        return (self.current_time, pkg.dst)

    def act(self, state):
        d = state[1]
        return self._mkBestEstimate(d)[0]

    def observe(self, sample):
        (d, y, new_estimate) = sample
        period = self.current_time - self.U[d][y]
        if period != 0:
            delta_q = new_estimate - self.Q[d][y]
            self.Q[d][y] += self.learning_rate * delta_q
            self.B[d][y] = min(self.B[d][y], self.Q[d][y])
            if delta_q < 0:
                delta_r = delta_q / period
                self.R[d][y] += self.beta_rate * delta_r
            elif delta_q > 0:
                self.R[d][y] *= self.gamma_rate
            self.U[d][y] = self.current_time

    def _currentStateData(self, pkg):
        return [self.current_time, pkg.id]

    def _currentStateCols(self):
        return ['time', 'pkg_id']


def dict_min(dct):
    return min(dct.items(), key=lambda x:x[1])

def mk_num_list(s, n):
    return list(map(lambda k: s+str(k), range(0, n)))

def mk_unary_arr(n, *pts):
    res = np.zeros(n)
    for p in pts:
        res[p] = 1
    return res
