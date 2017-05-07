from functools import total_ordering
import datetime as dt
import pandas as pd

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

    def __init__(self, graph, settings, results_file, logfile, **kwargs):
        super().__init__(**kwargs)
        self.graph = graph
        self.settings = settings
        self.results_file = results_file
        self.logfile = logfile

class SynchronizerInitMsg(InitMsg):
    """Init message for synchronizer"""

    def __init__(self, targets, delta=1.0, period=dt.timedelta(seconds=1), **kwargs):
        super().__init__(**kwargs)
        self.targets = targets
        self.delta = delta
        self.period = period

class PkgSenderInitMsg(InitMsg):
    """Init message for package sender"""

    def __init__(self, pkg_distr, sync_delta, network, **kwargs):
        super().__init__(**kwargs)
        self.pkg_distr = pkg_distr
        self.sync_delta = sync_delta
        self.network = network

class RouterInitMsg(InitMsg):
    """Init message for router"""

    def __init__(self, network_addr, neighbors, network, pkg_process_delay, **kwargs):
        super().__init__(**kwargs)
        self.network_addr = network_addr
        self.neighbors = neighbors
        self.network = network
        self.pkg_process_delay = pkg_process_delay

class SimpleQRouterInitMsg(RouterInitMsg):
    """Init message for SimpleQRouter"""

    def __init__(self, learning_rate, **kwargs):
        super().__init__(**kwargs)
        self.learning_rate = learning_rate

class DQNRouterInitMsg(RouterInitMsg):
    """Init message for DQNRouter"""

    def __init__(self, model_file, **kwargs):
        super().__init__(**kwargs)
        self.model_file = model_file

class LinkStateInitMsg(RouterInitMsg):
    """Init message for LinkStateRouter"""

# Routing messages
# ===
#
# These classes represent layer of information on routing level

class IncomingPkgEvent(EventMsg):
    pass

class ProcessPkgEvent(EventMsg):
    pass

class PkgTransferEndEvent(EventMsg):
    pass

class PkgDoneMsg(EventMsg):
    pass

class LinkBreakMsg(EventMsg):
    def __init__(self, time, sender, neighbor):
        super().__init__(time, sender, None)
        self.neighbor = neighbor

class LinkRestoreMsg(EventMsg):
    def __init__(self, time, sender, neighbor):
        super().__init__(time, sender, None)
        self.neighbor = neighbor

class RewardMsg(ServiceMsg):
    def __init__(self, pkg_id, **kwargs):
        super().__init__(kwargs)
        self.pkg_id = pkg_id

class SimpleRewardMsg(RewardMsg):
    def __init__(self, pkg_id, cur_time, estimate, dst, **kwargs):
        super().__init__(pkg_id, **kwargs)
        self.cur_time = cur_time
        self.estimate = estimate
        self.dst = dst

class DQNRewardMsg(RewardMsg):
    def __init__(self, pkg_id, pkg_size, cur_time, estimate, dst, **kwargs):
        super().__init__(pkg_id, **kwargs)
        self.pkg_size = pkg_size
        self.cur_time = cur_time
        self.estimate = estimate
        self.dst = dst

class LinkStateAnnouncement(ServiceMsg):
    def __init__(self, seq_num, from_addr, neighbors):
        self.seq_num = seq_num
        self.from_addr = from_addr
        self.neighbors = neighbors

@total_ordering
class Package:
    def __init__(self, pkg_id, size, dst, start_time, contents):
        self.id = pkg_id
        self.size = size
        self.dst = dst
        self.start_time = start_time
        self.route = None
        self.contents = contents

    def route_add(self, data, cols):
        if self.route is None:
            self.route = pd.DataFrame(columns=cols)
        self.route.loc[len(self.route)] = data

    def __eq__(self, other):
        return self.id == other.id

    def __lt__(self, other):
        return self.id < other.id

# Finishing job
# ===
#
# These are messages which are sent to/from overlord in the end

class ReportRequest(Message):
    pass

class ReportDone(Message):
    pass
