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
        self.neighbors = []
        self.package_id = 0

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
        if isinstance(event, PackageMsg):
            self.receivePackage(event)
            pkg = event.getContents()
            pkg.route_add(self.addr)
            if pkg.dst == self.addr:
                self.reportPkgDone(event)
            else:
                self.routePackage(event)
        else:
            pass

    def isInitialized(self):
        return self.overlord is not None

    def reportPkgDone(self, pkg_event):
        self.send(self.overlord, pkg_event)

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
                for k in self.neighbors:
                    if k == n:
                        self.Q[n][k] = 1
                    else:
                        self.Q[n][k] = len(self.network)

    def receivePackage(self, pkg_event):
        pkg = pkg_event.getContents()
        d = pkg.dst
        best_estimate = 0 if self.addr == d else dict_min(self.Q[d])[1]
        self.sendServiceMsg(pkg_event.sender, RewardMsg(id(pkg), self.current_time, best_estimate, dst))

    def routePackage(self, pkg_event):
        pkg = pkg_event.getContents()
        d = pkg.dst
        best_neighbor = dict_min(self.Q[d])[0]
        target = self.network[best_neighbor]
        self.routed_pkgs_times[id(pkg)] = self.current_time
        self.resendEventDelayed(target, pkg_event, delay)

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
