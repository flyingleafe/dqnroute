from functools import total_ordering
import datetime as dt

class Message:
    """Base class for all messages used in the system"""

    def __init__(self, contents):
        self.contents = contents

    def get_contents(self):
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

class OverlordInitMsg(InitMsg):
    """Init message for overlord"""

    def __init__(self, graph, packets_distr, emulation_settings, **kwargs):
        self.graph = graph
        self.packets_distr = packets_distr
        self.emulation_settings = emulation_settings
        super().__init__(kwargs)

class SynchronizerInitMsg(InitMsg):
    """Init message for synchronizer"""

    def __init__(self, targets, delta=1., period=dt.timedelta(seconds=1) **kwargs):
        self.targets = targets
        self.period = period
        super().__init__(kwargs)

class PkgSenderInitMsg(InitMsg):
    """Init message for package sender"""

    def __init__(self, n_packages, pkg_delta, sync_delta, network, **kwargs):
        self.n_packages = n_packages
        self.pkg_delta = pkg_delta
        self.sync_delta = sync_delta
        self.network = network
        super().__init__(kwargs)

class RouterInitMsg(InitMsg):
    """Init message for router"""

    def __init__(self, n, neighbors, network, **kwargs):
        self.network_addr = n
        self.neighbors = neighbors
        self.network = network
        super().__init__(kwargs)

# Routing messages
# ===
#
# These classes represent layer of information on routing level

class PackageMsg(EventMsg):
    pass

class RewardMsg(ServiceMsg):
    pass

class Package:
    def __init__(self, dst, start_time, contents):
        self.dst = dst
        self.start_time = start_time
        self.route = []
        self.contents = contents

    def route_add(self, addr):
        self.route.append(addr)
