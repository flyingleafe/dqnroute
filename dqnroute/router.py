import networkx as nx

from thespian.actors import *

from rl_agent import RLAgent
from messages import *
from time_actor import *

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
        self.link_states = {}

    def initialize(self, message, sender):
        super().initialize(message, sender)
        if isinstance(message, RouterInitMsg):
            self.overlord = sender
            self.addr = message.network_addr
            self.neighbors = message.neighbors
            self.network = message.network
            self.pkg_process_delay = message.pkg_process_delay
            self.network_inv = {str(target) : addr for addr, target in message.network.items()}
            for n in self.neighbors.keys():
                self.link_states[n] = {'transfer_time': 0, 'alive': True}

    def processEvent(self, event):
        if not self.isInitialized():
            raise RouterNotInitialized("Router has not been initialized!")

        if isinstance(event, IncomingPkgEvent):
            self.queue_time = max(self.current_time, self.queue_time) + self.pkg_process_delay
            new_event = ProcessPkgEvent(self.queue_time, event.sender, event.getContents())
            self.event_queue.push(new_event)
        elif isinstance(event, ProcessPkgEvent):
            self.receivePackage(event)
            pkg = event.getContents()
            pkg.route_add(self.addr)
            print("ROUTER #{} ROUTES PACKAGE TO {}".format(self.addr, pkg.dst))
            if pkg.dst == self.addr:
                self.reportPkgDone(pkg, self.current_time)
            else:
                best_neighbor = self.routePackage(pkg)
                target = self.network[best_neighbor]
                link_latency = self.neighbors[best_neighbor]['latency']
                link_bandwidth = self.neighbors[best_neighbor]['bandwidth']
                transfer_start_time = max(self.current_time, self.link_states[best_neighbor]['transfer_time'])
                transfer_end_time = transfer_start_time + (pkg.size / link_bandwidth)
                finish_time = transfer_end_time + link_latency
                self.link_states[best_neighbor]['transfer_time'] = transfer_end_time
                self.sendEvent(target, IncomingPkgEvent(finish_time, self.myAddress, pkg))
        else:
            pass

    def isInitialized(self):
        return self.overlord is not None

    def reportPkgDone(self, pkg, time):
        self.send(self.overlord, PkgDoneMsg(time, self.myAddress, pkg))

    def receivePackage(self, pkg_event):
        pass

    def routePackage(self, pkg_event):
        pass

class LinkStateRouter(Router):
    def __init__(self):
        super().__init__()
        self.seq_num = 0
        self.announcements = {}
        self.network_graph = None

    def initialize(self, message, sender):
        super().initialize(message, sender)
        if isinstance(message, LinkStateInitMsg):
            self.network_graph = nx.Graph()
            for n in self.network.keys():
                self.network_graph.add_node(n)
            for (n, data) in self.neighbors.items():
                alive = self.link_states[n]['alive']
                if alive:
                    self.network_graph.add_edge(self.addr, n, weight=data['latency'])
                elif self.network_graph.has_edge(self.addr, n):
                    self.network_graph.remove_edge(self.addr, n)
            self._announceLinkState()

    def _announceLinkState(self):
        neighbors_data = dict(self.network_graph.adjacency_iter())[self.addr]
        announcement = LinkStateAnnouncement(self.seq_num, self.addr, neighbors_data)
        self.announcements[self.addr] = (self.seq_num, neighbors_data)
        self._broadcastAnnouncement(announcement, self.myAddress)
        self.seq_num += 1

    def _broadcastAnnouncement(self, announcement, sender):
        for n in self.neighbors.keys():
            if sender != self.network[n]:
                self.sendServiceMsg(self.network[n], announcement)

    def receiveServiceMsg(self, message, sender):
        if isinstance(message, LinkStateAnnouncement):
            from_addr = message.from_addr
            seq = message.seq_num
            data = message.neighbors
            if from_addr not in self.announcements or self.announcements[from_addr][0] < seq:
                self.announcements[from_addr] = (seq, data)
                for (m, params) in data.items():
                    m_data = self.announcements.get(m, (0, {}))[1]
                    if from_addr in m_data:
                        self.network_graph.add_edge(from_addr, m, **params)
                for m in set(self.network.keys()) - set(data.keys()):
                    if self.network_graph.has_edge(from_addr, m):
                        self.network_graph.remove_edge(from_addr, m)

                self._broadcastAnnouncement(message, sender)

    def isInitialized(self):
        return super().isInitialized() and (len(self.announcements) == len(self.network))

    def routePackage(self, pkg):
        d = pkg.dst
        path = nx.shortest_path(self.network_graph, self.addr, d)
        return path[1]

class QRouter(Router, RLAgent):
    def __init__(self):
        super().__init__()
        self.reward_pending = {}

    def receivePackage(self, pkg_event):
        self.sendServiceMsg(pkg_event.sender, self.mkRewardMsg(pkg_event.getContents()))

    def routePackage(self, pkg):
        state = self.getState(pkg)
        self.reward_pending[pkg.id] = state
        return self.act(state)

    def receiveServiceMsg(self, message, sender):
        if isinstance(message, RewardMsg):
            prev_state = 0
            try:
                prev_state = self.reward_pending[message.pkg_id]
            except KeyError:
                print("Unexpected reward msg!")
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
    def __init__(self):
        super().__init__()
        self.Q = {}
        self.learning_rate = None
        self.steps = 0

    def initialize(self, message, sender):
        super().initialize(message, sender)
        if isinstance(message, SimpleQRouterInitMsg):
            self.learning_rate = message.learning_rate
            for n in self.network.keys():
                self.Q[n] = {}
                for (k, data) in self.neighbors.items():
                    if k == n:
                        self.Q[n][k] = 40
                    else:
                        self.Q[n][k] = 100500

    def mkRewardMsg(self, pkg):
        d = pkg.dst
        best_estimate = 0 if self.addr == d else dict_min(self.Q[d])[1]
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
        return dict_min(self.Q[d])[0]

    def observe(self, sample):
        (dst, sender_addr, new_estimate) = sample
        delta = self.learning_rate * (new_estimate - self.Q[dst][sender_addr])
        self.Q[dst][sender_addr] += delta

def dict_min(dct):
    return min(dct.items(), key=lambda x:x[1])
