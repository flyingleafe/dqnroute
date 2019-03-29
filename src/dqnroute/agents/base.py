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

    def currentTime(self) -> int:
        return self.env.time

    def handle(self, msg: Message) -> List[Message]:
        if isinstance(msg, InMessage):
            sender = msg.sender
            msg = msg.inner_msg

            if isinstance(msg, PkgMessage):
                pkg = msg.pkg
                if pkg.dst == self.id:
                    return [PkgReceivedMessage(pkg)]
                else:
                    to_interface, additional_msgs = self.route(sender, pkg)
                    logger.debug('Routing pkg #{} on router {} to router {}'.format(pkg.id, self.id, to_interface))
                    return [OutMessage(to_interface, PkgMessage(pkg))] + additional_msgs

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

class RewardAgent(object):
    def __init__(self, **kwargs):
        self._pending_pkgs = {}

    def registerSentPkg(self, pkg: Package, data):
        self._pending_pkgs[pkg.id] = data

    def receiveReward(self, msg: RewardMsg):
        saved_data = self._pending_pkgs.pop(msg.action_id)
        reward = self.computeReward(msg.reward_data, saved_data)
        return reward, saved_data

    def computeReward(self, observation, action, reward_data):
        raise NotImplementedError()

class NetworkRewardAgent(RewardAgent):
    def computeReward(self, reward_data, saved_data):
        time_received, Q = reward_data
        time_sent = saved_data['time_sent']
        return Q + (time_received - time_sent)

class ConveyorRewardAgent(RewardAgent):
    def computeReward(self, reward_data, saved_data):
        raise NotImplementedError()
