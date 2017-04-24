from thespian.actors import *

from messages import *
from time_actor import *

class RouterNotInitialized(Exception):
    """
    This is raised then some message which is not `RouterInitMsg` comes first to router
    """

class Router(TimeActor):
    def __init__(self):
        self.overlord = None
        self.addr = None
        self.network = {}
        self.neighbors = []
        self.package_id = 0

    def initialize(self, message, sender):
        if isinstance(message, RouterInitMsg):
            self.overlord = sender
            self.addr = message.network_addr
            self.neighbors = message.neighbors
            self.network = message.network
        super().initialize(message, sender)

    def processEvent(self, event):
        if not self.isInitialized():
            raise RouterNotInitialized("Router has not been initialized!")
        if isinstance(event, PackageMsg):
            pkg = event.get_contents()
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

    def routePackage(self, pkg_event):
        pass

class SimpleQRouter(Router):
    def customInit(self, custom):
        self.Q = {}
        self.learning_rate = custom['learning_rate']
        for n in self.network.keys():
            self.Q[n] = {}
            for k in self.neighbors:
                if k == n:
                    self.Q[n][k] = 1
                else:
                    self.Q[n][k] = len(self.network)

    def routePackage(self, pkg, pkg_id, sender):
        dst = pkg.destination()
        pass
