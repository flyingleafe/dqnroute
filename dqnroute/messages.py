from functools import total_ordering
import datetime as dt
import pandas as pd
import numpy as np

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

    def __hash__(self):
        return hash((self.time, self.contents))

    def __eq__(self, other):
        return (self.time, hash(self.contents)) == (other.time, hash(other.contents))

    def __lt__(self, other):
        return (self.time, hash(self.contents)) < (other.time, hash(other.contents))

class EventMsg(TimedMessage):
    """Base class for all messages which are subject to time simulation"""

    def __init__(self, time, sender, contents):
        self.sender = sender
        super().__init__(time, contents)

class TickMsg(TimedMessage):
    """Message for system time synchronization"""

    def __init__(self, time):
        super().__init__(time, None)

class SelfPlannedMsg(TimedMessage):
    """Message class for various planned events"""

    def __init__(self, time):
        super().__init__(time, -1)

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

class FinishInitMsg(Message):
    def __init__(self, child_id, **kwargs):
        super().__init__(kwargs)
        self.child_id = child_id

class OverlordInitMsg(InitMsg):
    """Init message for overlord"""

    def __init__(self, settings, results_file, logfile, router_type, **kwargs):
        super().__init__(**kwargs)
        self.settings = settings
        self.results_file = results_file
        self.logfile = logfile
        self.router_type = router_type

class RoutersOverlordInitMsg(OverlordInitMsg):
    def __init__(self, graph, **kwargs):
        super().__init__(**kwargs)
        self.graph = graph

class ConveyorsOverlordInitMsg(OverlordInitMsg):
    def __init__(self, configuration, **kwargs):
        super().__init__(**kwargs)
        self.configuration = configuration

class SynchronizerInitMsg(InitMsg):
    """Init message for synchronizer"""

    def __init__(self, targets, delta=1.0, period=dt.timedelta(seconds=1), **kwargs):
        super().__init__(**kwargs)
        self.targets = targets
        self.delta = delta
        self.period = period

class PkgSenderInitMsg(InitMsg):
    """Init message for package sender"""

    def __init__(self, pkg_distr, sync_delta, network, state_size, **kwargs):
        super().__init__(**kwargs)
        self.pkg_distr = pkg_distr
        self.sync_delta = sync_delta
        self.network = network
        self.state_size = state_size

class BagSenderInitMsg(PkgSenderInitMsg):
    """Init message for bag sender"""

    def __init__(self, sources, sinks, **kwargs):
        super().__init__(**kwargs)
        self.sources = sources
        self.sinks = sinks

class AbstractRouterInitMsg(InitMsg):
    """Init message for a node in some network"""

    def __init__(self, network_addr, network, full_log, **kwargs):
        super().__init__(**kwargs)
        self.network_addr = network_addr
        self.network = network
        self.full_log = full_log

class RouterInitMsg(AbstractRouterInitMsg):
    """Init message for network router"""

    def __init__(self, pkg_process_delay, neighbors, **kwargs):
        super().__init__(**kwargs)
        self.neighbors = neighbors
        self.pkg_process_delay = pkg_process_delay

class ConveyorInitMsg(AbstractRouterInitMsg):
    """Init message for conveyor belt"""

    def __init__(self, sections, all_sections, sinks, speed,
                 energy_consumption, stop_delay,
                 energy_reward_weight, sec_process_time, **kwargs):
        super().__init__(**kwargs)
        self.sections = sections
        self.all_sections = all_sections
        self.sinks = sinks
        self.speed = speed
        self.energy_consumption = energy_consumption
        self.stop_delay = stop_delay
        self.energy_reward_weight = energy_reward_weight
        self.sec_process_time = sec_process_time

class RouterFinalizeInitMsg(InitMsg):
    """Finalize init message for router"""

# Init messages for particular router types

class SimpleQRouterInitMsg(RouterInitMsg):
    """Init message for SimpleQRouter"""

    def __init__(self, learning_rate, **kwargs):
        super().__init__(**kwargs)
        self.learning_rate = learning_rate

class PredictiveQRouterInitMsg(RouterInitMsg):
    """Init message for PredictiveQRouter"""

    def __init__(self, learning_rate, beta_rate, gamma_rate, **kwargs):
        super().__init__(**kwargs)
        self.learning_rate = learning_rate
        self.beta_rate = beta_rate
        self.gamma_rate = gamma_rate

class DQNRouterInitMsg(RouterInitMsg):
    """Init message for DQNRouter"""

    def __init__(self, nn_type, batch_size=1, mem_capacity=1,
                 prioritized_xp=False, pkg_states=False, **kwargs):
        super().__init__(**kwargs)
        self.nn_type = nn_type
        self.mem_capacity = mem_capacity
        self.batch_size = batch_size
        self.prioritized_xp = prioritized_xp
        self.pkg_states = pkg_states

class LinkStateInitMsg(RouterInitMsg):
    """Init message for LinkStateRouter"""

# Init messages for particular conveyor types

class SimpleQConveyorInitMsg(ConveyorInitMsg):
    """Init message for SimpleQConveyor"""

    def __init__(self, learning_rate, **kwargs):
        super().__init__(**kwargs)
        self.learning_rate = learning_rate

class LinkStateConveyorInitMsg(ConveyorInitMsg):
    """Init message for link-state conveyor"""

class DQNConveyorInitMsg(ConveyorInitMsg):
    """Init message for DQNConveyor"""

    def __init__(self, nn_type, batch_size=1, mem_capacity=1,
                 prioritized_xp=False, pkg_states=False, **kwargs):
        super().__init__(**kwargs)
        self.nn_type = nn_type
        self.mem_capacity = mem_capacity
        self.batch_size = batch_size
        self.prioritized_xp = prioritized_xp
        self.pkg_states = False

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

# For conveyors
class IncomingLuggageEvent(EventMsg):
    def __init__(self, time, sender, section, prev_section, contents):
        super().__init__(time, sender, contents)
        self.section = section
        self.prev_section = prev_section

class ProcessLuggageEvent(EventMsg):
    def __init__(self, time, sender, section, prev_section, contents):
        super().__init__(time, sender, contents)
        self.section = section
        self.prev_section = prev_section
        self.prev_belt_stop_time = 0

class BagDoneMsg(PkgDoneMsg):
    pass

class LinkBreakMsg(EventMsg):
    def __init__(self, time, sender, neighbor):
        super().__init__(time, sender, None)
        self.neighbor = neighbor

class LinkRestoreMsg(EventMsg):
    def __init__(self, time, sender, neighbor):
        super().__init__(time, sender, None)
        self.neighbor = neighbor

class PkgReturnedMsg(ServiceMsg):
    pass

class BeltStatusMsg(ServiceMsg):
    def __init__(self, sections, working=False):
        self.sections = sections
        self.working = working

# Reward messages
# ===
#
# For RL agents

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

class PredictiveRewardMsg(RewardMsg):
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

# For conveyors

class ConveyorRewardMsg(ServiceMsg):
    def __init__(self, bag_id, section, **kwargs):
        super().__init__(kwargs)
        self.bag_id = bag_id
        self.section = section

class SimpleQConveyorRewardMsg(ConveyorRewardMsg):
    def __init__(self, bag_id, section, reward, estimate, **kwargs):
        super().__init__(bag_id, section, **kwargs)
        self.reward = reward
        self.estimate = estimate

class DQNConveyorRewardMsg(ConveyorRewardMsg):
    def __init__(self, bag_id, section, reward, estimate, **kwargs):
        super().__init__(bag_id, section, **kwargs)
        self.reward = reward
        self.estimate = estimate

# Other messages
# ===

class LinkStateAnnouncement(ServiceMsg):
    def __init__(self, seq_num, from_addr, neighbors):
        self.seq_num = seq_num
        self.from_addr = from_addr
        self.neighbors = neighbors

class ConveyorLinkStateAnnouncement(ServiceMsg):
    def __init__(self, seq_num, section_infos):
        self.seq_num = seq_num
        self.section_infos = section_infos

class NeighborsAdvice(ServiceMsg):
    def __init__(self, time, estimations):
        self.time = time
        self.estimations = estimations

class NeighborLoadStatus(ServiceMsg):
    def __init__(self, is_overloaded):
        self.is_overloaded = is_overloaded

# Planned events for conveyors
# ===

class StopMovingEvent(SelfPlannedMsg):
    pass

@total_ordering
class Package:
    def __init__(self, pkg_id, size, dst, start_time, state_size, contents):
        self.id = pkg_id
        self.size = size
        self.dst = dst
        self.start_time = start_time
        self.route = None
        self.contents = contents
        self.rnn_state = (np.zeros((1, state_size)),
                          np.zeros((1, state_size)))

    def route_add(self, data, cols):
        if self.route is None:
            self.route = pd.DataFrame(columns=cols)
        self.route.loc[len(self.route)] = data

    def __hash__(self):
        return hash((self.id, self.contents))

    def __eq__(self, other):
        return self.id == other.id

    def __lt__(self, other):
        return self.id < other.id

class Bag(Package):
    def __init__(self, bag_id, dst, start_time, prev_time, state_size, contents):
        super().__init__(bag_id, 0, dst, start_time, state_size, contents)
        self.prev_time = prev_time
        self.energy_spent = 0

# Finishing job
# ===
#
# These are messages which are sent to/from overlord in the end

class ReportRequest(Message):
    pass

class ReportDone(Message):
    pass
