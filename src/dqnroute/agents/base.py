import logging

from typing import List, Tuple
from ..messages import *
from ..utils import *
from ..constants import DQNROUTE_LOGGER
from ..time_env import TimeEnv

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
    def __init__(self, time_env: TimeEnv, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.time_env = time_env

    def currentTime(self) -> int:
        return self.time_env.currentTime()

    def handle(self, msg: Message) -> List[Message]:
        if isinstance(msg, InMessage):
            sender = msg.sender
            msg = msg.getContents()

            if isinstance(msg, PkgMessage):
                pkg = msg.getContents()
                if pkg.dst == self.id:
                    return [PkgReceivedMessage(pkg)]
                else:
                    to_interface, additional_msgs = self.route(sender, pkg)
                    logger.debug('Routing pkg #{} on router {} to router {}'.format(pkg.id, self.id, to_interface))
                    return [OutMessage(to_interface, PkgMessage(pkg))] + additional_msgs

            elif isinstance(msg, ServiceMessage):
                return self.handleServiceMsg(sender, msg)

        elif isinstance(msg, InitMessage):
            return self.init(msg.getContents())

        elif isinstance(msg, AddLinkMessage):
            return self.addLink(**msg.getContents())

        elif isinstance(msg, RemoveLinkMessage):
            return self.removeLink(msg.getContents())

        else:
            return super().handle(msg)

    def init(self, config) -> List[Message]:
        self.id = config["id"]
        self.neighbour_ids = set(config["neighbour_ids"])
        return []

    def addLink(self, to: int, params={}) -> List[Message]:
        self.neighbour_ids.add(to)
        return []

    def removeLink(self, to: int) -> List[Message]:
        self.neighbour_ids.remove(to)
        return []

    def route(self, sender: int, pkg: Package) -> Tuple[int, List[Message]]:
        raise NotImplementedError()

    def handleServiceMsg(self, sender: int, msg: ServiceMessage) -> List[Message]:
        raise UnsupportedMessageType(msg)

class RewardAgent(object):
    def __init__(self):
        self._pending_pkgs = {}

    def registerSentPkg(self, pkg: Package, data):
        self._pending_pkgs[pkg.id] = data

    def receiveReward(self, msg: RewardMsg):
        args = msg.getContents()
        pkg_id = args['action_id']
        reward_data = args['reward_data']

        saved_data = self._pending_pkgs.pop(pkg_id)
        reward = self.computeReward(reward_data, saved_data)
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
