from functools import total_ordering
import datetime as dt

class Message:
    """Base class for all messages used in the system"""

    def __init__(self, contents):
        self.contents = contents

    def getContents(self):
        return self.contents

# Message classes for events layer
# ===
#
# These classes are the basis for time simulation

@total_ordering
class TimedMessage(Message):
    """Class of messages which can be compared by time"""

    def __init__(self, time, contents):
        self.time = time
        super().__init__(contents)

    def __eq__(self, other):
        return (self.time, self.contents) == (other.time, other.contents)

    def __lt__(self, other):
        return (self.time, self.contents) < (other.time, other.contents)

class EventMsg(TimedMessage):
    """Base class for all messages which are subject to time simulation"""

    def __init__(self, time, sender, contents):
        self.sender = sender
        super().__init__(time, contents)

class TickMsg(TimedMessage):
    """Message for system time synchronization"""

    def __init__(self, time):
        super().__init__(time, None)

class ServiceMsg(Message):
    """Any message which doesn't take part in time simulation"""

# Initialization messages
# ===
#
# These classes represent service messages which are used on system init

class InitMsg(Message):
    """Base class for init message (to distinguish it from every other)"""
    def __init__(self, **kwargs):
        super().__init__(kwargs)

class OverlordInitMsg(InitMsg):
    """Init message for overlord"""

    def __init__(self, graph, packets_distr, emulation_settings, **kwargs):
        super().__init__(**kwargs)
        self.graph = graph
        self.packets_distr = packets_distr
        self.emulation_settings = emulation_settings

class SynchronizerInitMsg(InitMsg):
    """Init message for synchronizer"""

    def __init__(self, targets, delta=1.0, period=dt.timedelta(seconds=1), **kwargs):
        super().__init__(**kwargs)
        self.targets = targets
        self.delta = delta
        self.period = period

class PkgSenderInitMsg(InitMsg):
    """Init message for package sender"""

    def __init__(self, n_packages, pkg_delta, sync_delta, network, **kwargs):
        super().__init__(**kwargs)
        self.n_packages = n_packages
        self.pkg_delta = pkg_delta
        self.sync_delta = sync_delta
        self.network = network

class RouterInitMsg(InitMsg):
    """Init message for router"""

    def __init__(self, network_addr, neighbors, network, **kwargs):
        super().__init__(**kwargs)
        self.network_addr = network_addr
        self.neighbors = neighbors
        self.network = network

class QRouterInitMsg(RouterInitMsg):
    """Init message for QRouter"""

    def __init__(self, learning_rate, **kwargs):
        super().__init__(**kwargs)
        self.learning_rate = learning_rate

# Routing messages
# ===
#
# These classes represent layer of information on routing level

class PackageMsg(EventMsg):
    pass

class RewardMsg(ServiceMsg):
    def __init__(self, pkg_id, cur_time, estimate, dst, **kwargs):
        super().__init__(kwargs)
        self.pkg_id = pkg_id
        self.cur_time = cur_time
        self.estimate = estimate
        self.dst = dst

@total_ordering
class Package:
    def __init__(self, pkg_id, dst, start_time, contents):
        self.id = pkg_id
        self.dst = dst
        self.start_time = start_time
        self.route = []
        self.contents = contents

    def route_add(self, addr):
        self.route.append(addr)

    def __eq__(self, other):
        return self.id == other.id

    def __lt__(self, other):
        return self.id < other.id
