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
        self.pkg_process_delay = 3
        self.queue_time = 0
        self.link_states = {}

    def initialize(self, message, sender):
        super().initialize(message, sender)
        if isinstance(message, RouterInitMsg):
            self.overlord = sender
            self.addr = message.network_addr
            self.neighbors = message.neighbors
            self.network = message.network
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

class SimpleQRouter(QRouter):
    def __init__(self):
        super().__init__()
        self.Q = {}
        self.learning_rate = None

    def initialize(self, message, sender):
        super().initialize(message, sender)
        if isinstance(message, SimpleQRouterInitMsg):
            self.learning_rate = message.learning_rate
            for n in self.network.keys():
                self.Q[n] = {}
                for (k, data) in self.neighbors.items():
                    if k == n:
                        self.Q[n][k] = 2 * data['latency']
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
