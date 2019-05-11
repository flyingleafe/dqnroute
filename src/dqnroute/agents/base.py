import logging

from typing import List, Tuple
from ..messages import *
from ..utils import *
from ..constants import DQNROUTE_LOGGER

logger = logging.getLogger(DQNROUTE_LOGGER)

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

    def _wireMsg(self, event: WorldEvent) -> WorldEvent:
        if isinstance(event, Message):
            if isinstance(event, DelayInterruptMessage):
                return event
            elif isinstance(event, DelayedMessage):
                return DelayedMessage(event.id, event.delay, self._wireMsg(event.inner_msg))
            elif isinstance(event, OutMessage) and event.from_node == self.id:
                return WireOutMsg(self.interface_inv_map[event.to_node], event.inner_msg)
            else:
                return WireOutMsg(-1, event)
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

        return [self._wireMsg(m) for m in evs]

    def handleMsg(self, msg: Message) -> List[WorldEvent]:
        if isinstance(msg, InitMessage):
            return self.init(msg.config)

        elif isinstance(msg, InMessage):
            assert self.id == msg.to_node, \
                "Wrong recipient of InMessage!"
            return self.handleMsgFrom(msg.from_node, msg.inner_msg)

        else:
            raise UnsupportedMessageType(msg)

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
        raise UnsupportedEventType(event)


class Router(MessageHandler):
    """
    Agent which routes packages and service messages.
    """
    def __init__(self, out_neighbours: List[AgentId], in_neighbours: List[AgentId], **kwargs):
        super().__init__(**kwargs)
        self.out_neighbours = set(out_neighbours)
        self.in_neighbours = set(in_neighbours)

    def __getattr__(self, name):
        if name == 'all_neighbours':
            return self.in_neighbours | self.out_neighbours
        else:
            raise AttributeError(name)

    def handleEvent(self, event: WorldEvent) -> List[WorldEvent]:
        if isinstance(event, PkgEnqueuedEvent):
            assert event.recipient == self.id, \
                "Wrong recipient of PkgEnqueuedEvent!"
            return self.detectEnqueuedPkg()

        elif isinstance(event, PkgProcessingEvent):
            assert event.recipient == self.id, \
                "Wrong recipient of PkgProcessingEvent!"

            pkg = event.pkg
            sender = event.sender
            if pkg.dst == self.id:
                return [PkgReceiveAction(pkg)]
            else:
                to_nbr, additional_msgs = self.route(sender, pkg)
                logger.debug('Routing pkg #{} on router {} to router {}'.format(pkg.id, self.id[1], to_nbr[1]))
                return [PkgRouteAction(to_nbr, pkg)] + additional_msgs

        elif isinstance(event, LinkUpdateEvent):
            assert self.id in [event.u, event.v], \
                "Wrong recipient of LinkUpdateEvent!"

            other = event.u if event.u != self.id else event.v

            if isinstance(event, AddLinkEvent):
                return self.addLink(other, event.params)

            elif isinstance(event, RemoveLinkEvent):
                return self.removeLink(other)

        else:
            return super().handleEvent(event)

    def detectEnqueuedPkg(self) -> List[WorldEvent]:
        return []

    def addLink(self, other: AgentId, params={}) -> List[WorldEvent]:
        self.in_neighbours.add(other)
        self.out_neighbours.add(other)
        return []

    def removeLink(self, other: AgentId) -> List[WorldEvent]:
        self.in_neighbours.remove(other)
        self.out_neighbours.remove(other)
        return []

    def route(self, sender: AgentId, pkg: Package) -> Tuple[AgentId, List[Message]]:
        """
        Subclasses should reimplement this
        """
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
        else:
            return super().handleMsgFrom(sender, msg)

    def start(self) -> List[WorldEvent]:
        raise NotImplementedError()

    def stop(self) -> List[WorldEvent]:
        raise NotImplementedError()

    def handleIncomingBag(self, notifier: AgentId, bag: Bag) -> List[WorldEvent]:
        raise NotImplementedError()

    def handleOutgoingBag(self, notifier: AgentId, bag: Bag) -> List[WorldEvent]:
        raise NotImplementedError()


class Diverter(MessageHandler):
    """
    Base class which implements a diverter controller, which detects
    incoming bags and either kicks them out or not.
    """
    def __init__(self, host_conveyor: AgentId, **kwargs):
        super().__init__(**kwargs)
        self.host_conveyor = host_conveyor

    def handleEvent(self, event: WorldEvent) -> List[WorldEvent]:
        if isinstance(event, BagDetectionEvent):
            kick, msgs = self.divert(event.bag)
            if kick:
                msgs.append(DiverterKickAction())
            return msgs
        else:
            raise UnsupportedEventType(event)

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
