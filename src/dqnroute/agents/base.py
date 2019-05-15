import logging

from copy import deepcopy
from typing import List, Tuple, Callable
from ..messages import *
from ..utils import *
from ..constants import DQNROUTE_LOGGER

logger = logging.getLogger(DQNROUTE_LOGGER)


class BrokenInterfaceError(Exception):
    """
    Thrown when a message is sent through non-existing (broken) interface
    """
    def __init__(self, nbr: AgentId, msg: Message):
        self.nbr = nbr
        self.msg = msg


class EventHandler(object):
    """
    An abstract class which handles `WorldEvent`s and produces other `WorldEvent`s.
    """
    def __init__(self, **kwargs):
        # Simply ignores the parameters; done for convenience.
        super().__init__()

    def handle(self, event: WorldEvent) -> List[WorldEvent]:
        raise UnsupportedEventType(event)


class MessageHandler(EventHandler):
    """
    Abstract class for the agent which interacts with neighbours via messages.
    Performs mapping between outbound interfaces and neighbours IDs.
    """
    def __init__(self, id: AgentId, env: DynamicEnv, neighbours: List[AgentId], **kwargs):
        super().__init__(**kwargs)
        self.id = id
        self.env = env
        self.interface_map = {}
        self.interface_inv_map = {}

        for (i, n) in enumerate(neighbours):
            self.interface_map[i] = n
            self.interface_inv_map[n] = i

        # delays functionality
        self._delay_seq = 0
        self._pending_delayed = {}

        # lost messages
        self._lost_msgs = {}

    def _wireMsg(self, event: WorldEvent) -> WorldEvent:
        if isinstance(event, Message):
            if isinstance(event, OutMessage) and event.from_node == self.id:
                try:
                    return WireOutMsg(self.interface_inv_map[event.to_node], event.inner_msg)
                except KeyError:
                    raise BrokenInterfaceError(event.to_node, event.inner_msg)
            else:
                return WireOutMsg(-1, event)
        elif isinstance(event, DelayedEvent):
            return DelayedEvent(event.id, event.delay, self._wireMsg(event.inner))
        else:
            return event

    def _unwireMsg(self, msg: WireInMsg) -> Message:
        int_id = msg.interface
        inner = msg.payload

        if int_id == -1:
            return inner
        return InMessage(self.interface_map[int_id], self.id, inner)

    def handle(self, event: WorldEvent) -> List[WorldEvent]:
        if isinstance(event, WireInMsg):
            evs = self.handleMsg(self._unwireMsg(event))
        else:
            evs = self.handleEvent(event)

        res = []
        for m in evs:
            try:
                res.append(self._wireMsg(m))
            except BrokenInterfaceError as err:
                nbr = err.nbr
                if nbr not in self._lost_msgs:
                    self._lost_msgs[nbr] = []
                self._lost_msgs[nbr].append(err.msg)
        return res

    def handleMsg(self, msg: Message) -> List[WorldEvent]:
        if isinstance(msg, InitMessage):
            return self.init(msg.config)

        elif isinstance(msg, DelayTriggerMsg):
            callback = self._pending_delayed.pop(msg.delay_id)
            return callback()

        elif isinstance(msg, InMessage):
            assert self.id == msg.to_node, \
                "Wrong recipient of InMessage!"
            return self.handleMsgFrom(msg.from_node, msg.inner_msg)

        else:
            return self.handleMsgFrom(self.id, msg)

    def init(self, config) -> List[WorldEvent]:
        """
        Does nothing on initialization by default. Override if necessary.
        """
        return []

    def handleMsgFrom(self, sender: AgentId, msg: Message) -> List[WorldEvent]:
        """
        Should be overridden by subclasses
        """
        raise UnsupportedMessageType(msg)

    def handleEvent(self, event: WorldEvent) -> List[WorldEvent]:
        """
        Should be overridden by subclasses
        """
        if isinstance(event, InterfaceSetupEvent):
            iid = event.interface
            nbr = event.neighbour
            self.interface_map[iid] = nbr
            self.interface_inv_map[nbr] = iid
            found_msgs = [OutMessage(self.id, nbr, m) for m in self._lost_msgs.pop(nbr, [])]

            return found_msgs + self.addLink(nbr, event.params)
        elif isinstance(event, InterfaceShutdownEvent):
            iid = event.interface
            nbr = self.interface_map.pop(iid)
            del self.interface_inv_map[nbr]

            return self.removeLink(nbr)
        else:
            raise UnsupportedEventType(event)

    def addLink(self, other: AgentId, params={}) -> List[WorldEvent]:
        """
        Should be overridden by subclasses
        """
        return []

    def removeLink(self, other: AgentId) -> List[WorldEvent]:
        """
        Should be overridden by subclasses
        """
        return []

    def delayed(self, delay: float, callback: Callable[[], List[WorldEvent]]) -> DelayedEvent:
        """
        Schedule an action returning a list of `WorldEvent`s after some time.
        """
        delay_id = self._delay_seq
        self._delay_seq += 1
        self._pending_delayed[delay_id] = callback
        return DelayedEvent(delay_id, delay, DelayTriggerMsg(delay_id))

    def cancelDelayed(self, delay_id: int) -> DelayInterrupt:
        """
        Cancel a previously scheduled action before it's happened
        """
        del self._pending_delayed[delay_id]
        return DelayInterrupt(delay_id)

    def broadcast(self, msg: Message, exclude=[]) -> List[Message]:
        """
        Send a copy of a message to all neighbours, excluding given
        """
        return [OutMessage(self.id, v, deepcopy(msg))
                for v in (set(self.interface_map.values()) - set(exclude))]


class MasterHandler(MessageHandler):
    """
    A handler which controls all the nodes in the system in a centralized way.
    """
    def __init__(self, **kwargs):
        super().__init__(id=('master', 0), neighbours=[], **kwargs)
        self.slaves = {}

    def handleEvent(self, event: WorldEvent) -> List[WorldEvent]:
        if isinstance(event, SlaveEvent):
            return self.handleSlaveEvent(event.slave_id, event.inner)
        else:
            return super().handleEvent(event)

    def registerSlave(self, slave_id: AgentId, slave):
        self.slaves[slave_id] = slave

    def handleSlaveEvent(self, slave_id: AgentId, event: WorldEvent):
        raise UnsupportedEventType(event)


class SlaveHandler(MessageHandler):
    """
    Agent which is controlled in a centralized way via master handler.
    Should not be connected to anything.
    """
    def __init__(self, id: AgentId, master: MasterHandler):
        super().__init__(id=id, env=DynamicEnv(), neighbours=[])
        self.master = master

    def init(self, config):
        self.master.registerSlave(self.id, self)
        return []

    def handleEvent(self, event: WorldEvent) -> List[WorldEvent]:
        return self.master.handleEvent(SlaveEvent(self.id, event))

class Router(MessageHandler):
    """
    Agent which routes packages and service messages.
    """
    def handleEvent(self, event: WorldEvent) -> List[WorldEvent]:
        if isinstance(event, PkgEnqueuedEvent):
            assert event.recipient == self.id, \
                "Wrong recipient of PkgEnqueuedEvent!"
            return self.detectEnqueuedPkg()

        elif isinstance(event, PkgProcessingEvent):
            assert event.recipient == self.id, \
                "Wrong recipient of PkgProcessingEvent!"

            pkg = event.pkg
            if pkg.dst == self.id:
                return [PkgReceiveAction(pkg)]
            else:
                sender = event.sender
                allowed_nbrs = event.allowed_nbrs
                if allowed_nbrs is None:
                    allowed_nbrs = list(self.interface_map.values())

                to_nbr, additional_msgs = self.route(sender, pkg, allowed_nbrs)
                assert to_nbr in allowed_nbrs, "Resulting neighbour is not among allowed!"

                logger.debug('Routing pkg #{} on router {} to router {}'.format(pkg.id, self.id[1], to_nbr[1]))
                return [PkgRouteAction(to_nbr, pkg)] + additional_msgs

        else:
            return super().handleEvent(event)

    def detectEnqueuedPkg(self) -> List[WorldEvent]:
        return []

    def route(self, sender: AgentId, pkg: Package, allowed_nbrs: List[AgentId]) -> Tuple[AgentId, List[Message]]:
        """
        Subclasses should reimplement this
        """
        raise NotImplementedError()


class BagDetector(MessageHandler):
    """
    Base class for `Source`s and `Sink`s. Only reacts to `BagDetectionEvent`.
    """
    def handleEvent(self, event: WorldEvent) -> List[WorldEvent]:
        if isinstance(event, BagDetectionEvent):
            return self.bagDetection(event.bag)
        else:
            return super().handleEvent(event)

    def bagDetection(self, bag: Bag) -> List[WorldEvent]:
        raise NotImplementedError()


class Conveyor(MessageHandler):
    """
    Base class which implements a conveyor controller, which can start
    a conveyor or stop it.
    """
    def handleMsgFrom(self, sender: AgentId, msg: Message) -> List[WorldEvent]:
        if isinstance(msg, IncomingBagMsg):
            return self.handleIncomingBag(sender, msg.bag)
        elif isinstance(msg, OutgoingBagMsg):
            return self.handleOutgoingBag(sender, msg.bag)
        elif isinstance(msg, PassedBagMsg):
            return self.handlePassedBag(sender, msg.bag)
        else:
            return super().handleMsgFrom(sender, msg)

    def handleIncomingBag(self, notifier: AgentId, bag: Bag) -> List[WorldEvent]:
        raise NotImplementedError()

    def handleOutgoingBag(self, notifier: AgentId, bag: Bag) -> List[WorldEvent]:
        raise NotImplementedError()

    def handlePassedBag(self, notifier: AgentId, bag: Bag) -> List[WorldEvent]:
        raise NotImplementedError()


class Diverter(BagDetector):
    """
    Base class which implements a diverter controller, which detects
    incoming bags and either kicks them out or not.
    """
    def bagDetection(self, bag: Bag) -> List[WorldEvent]:
        kick, msgs = self.divert(bag)
        if kick:
            msgs.append(DiverterKickAction())
        return msgs

    def divert(self, bag: Bag) -> Tuple[bool, List[Message]]:
        raise NotImplementedError()


class RewardAgent(object):
    """
    Agent which receives rewards for sent packages
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._pending_pkgs = {}

    def registerResentPkg(self, pkg: Package, Q_estimate: float, action, data) -> RewardMsg:
        rdata = self._getRewardData(pkg, data)
        self._pending_pkgs[pkg.id] = (action, rdata, data)
        return self._mkReward(pkg.id, Q_estimate, rdata)

    def receiveReward(self, msg: RewardMsg):
        action, old_reward_data, saved_data = self._pending_pkgs.pop(msg.pkg_id)
        reward = self._computeReward(msg, old_reward_data)
        return action, reward, saved_data

    def _computeReward(self, msg: RewardMsg, old_reward_data):
        raise NotImplementedError()

    def _mkReward(self, pkg_id: int, Q_estimate: float, reward_data) -> RewardMsg:
        raise NotImplementedError()

    def _getRewardData(self, pkg: Package, data):
        raise NotImplementedError()

class NetworkRewardAgent(RewardAgent):
    """
    Agent which receives and processes rewards in computer networks.
    """

    def _computeReward(self, msg: NetworkRewardMsg, time_sent: float):
        time_received = msg.reward_data
        return msg.Q_estimate + (time_received - time_sent)

    def _mkReward(self, pkg_id: int, Q_estimate: float, time_sent: float) -> NetworkRewardMsg:
        return NetworkRewardMsg(pkg_id, Q_estimate, time_sent)

    def _getRewardData(self, pkg: Package, data):
        return self.env.time()

class ConveyorRewardAgent(RewardAgent):
    """
    Agent which receives and processes rewards in conveyor networks
    """

    def __init__(self, energy_reward_weight, **kwargs):
        super().__init__(**kwargs)
        self._e_weight = energy_reward_weight

    def _computeReward(self, msg: ConveyorRewardMsg, old_reward_data):
        time_sent, _ = old_reward_data
        time_processed, energy_gap = msg.reward_data
        time_gap = time_processed - time_sent
        return msg.Q_estimate + time_gap + self._e_weight * energy_gap

    def _mkReward(self, bag_id: int, Q_estimate: float, reward_data) -> ConveyorRewardMsg:
        time_processed, energy_gap = reward_data
        return ConveyorRewardMsg(bag_id, Q_estimate, time_processed, energy_gap)

    def _getRewardData(self, bag: Bag, data):
        cur_time = self.env.time()
        delay = self.env.stop_delay()
        consumption = self.env.energy_consumption()
        stop_time = self.env.scheduled_stop()
        time_gap = delay - max(0, stop_time - cur_time)
        energy_gap = consumption * time_gap

        return cur_time, energy_gap
