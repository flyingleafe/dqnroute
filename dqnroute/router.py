from thespian.actors import *

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
        self.pkg_process_delay = 0.2
        self.queue_time = 0

    def initialize(self, message, sender):
        super().initialize(message, sender)
        if isinstance(message, RouterInitMsg):
            self.overlord = sender
            self.addr = message.network_addr
            self.neighbors = message.neighbors
            self.network = message.network
            self.network_inv = {str(target) : addr for addr, target in message.network.items()}

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
                link_weight = self.neighbors[best_neighbor]['weight']
                finish_time = self.current_time + link_weight
                self.sendEvent(target, IncomingPkgEvent(finish_time, self.myAddress, pkg))
                self.event_queue.push(PkgTransferEndEvent(finish_time, self.myAddress, pkg))
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

class SimpleQRouter(Router):
    def __init__(self):
        super().__init__()
        self.Q = {}
        self.routed_pkgs_times = {}
        self.learning_rate = None

    def initialize(self, message, sender):
        super().initialize(message, sender)
        if isinstance(message, QRouterInitMsg):
            self.learning_rate = message.learning_rate
            for n in self.network.keys():
                self.Q[n] = {}
                for (k, data) in self.neighbors.items():
                    if k == n:
                        self.Q[n][k] = 1
                    else:
                        self.Q[n][k] = len(self.network)

    def receivePackage(self, pkg_event):
        pkg = pkg_event.getContents()
        d = pkg.dst
        best_estimate = 0 if self.addr == d else dict_min(self.Q[d])[1]
        self.sendServiceMsg(pkg_event.sender, RewardMsg(pkg.id, self.current_time, best_estimate, d))

    def routePackage(self, pkg):
        d = pkg.dst
        self.routed_pkgs_times[pkg.id] = self.current_time
        return dict_min(self.Q[d])[0]

    def receiveServiceMsg(self, message, sender):
        if isinstance(message, RewardMsg):
            sender_addr = self.network_inv[str(sender)]
            dst = message.dst
            sent_time = None
            try:
                sent_time = self.routed_pkgs_times[message.pkg_id]
            except KeyError:
                print("Unexpected reward msg!")
            new_estimate = message.estimate + (message.cur_time - sent_time)
            delta = self.learning_rate * (new_estimate - self.Q[dst][sender_addr])
            self.Q[dst][sender_addr] += delta

def dict_min(dct):
    return min(dct.items(), key=lambda x:x[1])
