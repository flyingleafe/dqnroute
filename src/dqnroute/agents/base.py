import logging

from typing import List, Tuple
from ..messages import *
from ..utils import *
from ..constants import DQNROUTE_LOGGER

logger = logging.getLogger(DQNROUTE_LOGGER)

class MessageHandler(object):
    """
    Abstract class for the piece of code which interacts with its
    environment via messages.
    """

    def handle(self, msg: Message) -> List[Message]:
        raise UnsupportedMessageType(msg)

class Router(MessageHandler):
    """
    Agent which routes packages and service messages.
    """
    def __init__(self, env: DynamicEnv, router_id: int,
                 out_neighbours: List[int], in_neighbours: List[int], **kwargs):
        super().__init__()
        self.env = env
        self.id = router_id
        self.out_neighbours = set(out_neighbours)
        self.in_neighbours = set(in_neighbours)

    def __getattr__(self, name):
        if name == 'all_neighbours':
            return self.in_neighbours | self.out_neighbours
        else:
            raise AttributeError(name)

    def handle(self, msg: Message) -> List[Message]:
        if isinstance(msg, InMessage):
            sender = msg.from_node
            msg = msg.inner_msg

            if isinstance(msg, PkgMessage):
                pkg = msg.pkg
                if pkg.dst == self.id:
                    return [PkgReceivedMessage(pkg)]
                else:
                    to_interface, additional_msgs = self.route(sender, pkg)
                    logger.debug('Routing pkg #{} on router {} to router {}'.format(pkg.id, self.id, to_interface))
                    return [OutMessage(self.id, to_interface, PkgMessage(pkg))] + additional_msgs

            elif isinstance(msg, ServiceMessage):
                return self.handleServiceMsg(sender, msg)

        elif isinstance(msg, InitMessage):
            return self.init(msg.config)

        elif isinstance(msg, AddLinkMessage):
            return self.addLink(**msg.getContents())

        elif isinstance(msg, RemoveLinkMessage):
            return self.removeLink(**msg.getContents())

        else:
            return super().handle(msg)

    def init(self, config) -> List[Message]:
        return []

    def addLink(self, to: int, direction: str, params={}) -> List[Message]:
        if direction != 'out':
            self.in_neighbours.add(to)
        if direction != 'in':
            self.out_neighbours.add(to)
        return []

    def removeLink(self, to: int, direction: str) -> List[Message]:
        if direction != 'out':
            self.in_neighbours.remove(to)
        if direction != 'in':
            self.out_neighbours.remove(to)
        return []

    def route(self, sender: int, pkg: Package) -> Tuple[int, List[Message]]:
        raise NotImplementedError()

    def handleServiceMsg(self, sender: int, msg: ServiceMessage) -> List[Message]:
        raise UnsupportedMessageType(msg)

class Conveyor(MessageHandler):
    """
    Base class which implements a conveyor controller, which can start
    a conveyor or stop it.
    """
    def __init__(self, env: DynamicEnv, conveyor_id: int, **kwargs):
        super().__init__()
        self.env = env
        self.id = conveyor_id

    def handle(self, msg: Message) -> List[Message]:
        if isinstance(msg, ConveyorStartMsg):
            return self.start()
        elif isinstance(msg, ConveyorStopMsg):
            return self.stop()
        elif isinstance(msg, IncomingBagMsg):
            return self.handleIncomingBag(msg.bag)
        elif isinstance(msg, OutgoingBagMsg):
            return self.handleOutgoingBag(msg.bag)
        elif isinstance(msg, ConveyorServiceMsg):
            return self.handleCustomMsg(msg)
        else:
            return super().handle(msg)

    def start(self) -> List[Message]:
        raise NotImplementedError()

    def stop(self) -> List[Message]:
        raise NotImplementedError()

    def handleIncomingBag(self, bag: Bag) -> List[Message]:
        raise NotImplementedError()

    def handleOutgoingBag(self, bag: Bag) -> List[Message]:
        raise NotImplementedError()

    def handleCustomMsg(self, msg: ConveyorMessage) -> List[Message]:
        raise UnsupportedMessageType(msg)

class RewardAgent(object):
    """
    Agent which receives rewards for sent packages
    """

    def __init__(self, **kwargs):
        self._pending_pkgs = {}

    def registerResentPkg(self, pkg: Package, Q_estimate: float, data) -> RewardMsg:
        rdata = self._getRewardData(pkg, data)
        self._pending_pkgs[pkg.id] = (rdata, data)
        return self._mkReward(pkg.id, Q_estimate, rdata)

    def receiveReward(self, msg: RewardMsg):
        old_reward_data, saved_data = self._pending_pkgs.pop(msg.pkg_id)
        reward = self._computeReward(msg, old_reward_data)
        return reward, saved_data

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

    def __init__(self, energy_reward_weight=1, **kwargs):
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
