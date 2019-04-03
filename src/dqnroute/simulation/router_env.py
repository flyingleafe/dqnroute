import logging
import math

from typing import List, Callable, Dict
from simpy import Environment, Event, Resource, Process
from ..utils import *
from ..messages import *
from ..event_series import EventSeries
from ..agents import Router
from ..constants import DQNROUTE_LOGGER
from .message_env import SimpyMessageEnv

logger = logging.getLogger(DQNROUTE_LOGGER)

class SimpyRouterEnv(SimpyMessageEnv):
    """
    Router environment which plays the role of outer world and
    handles message passing between routers.
    Passing a message to a router is done via `handle` method.
    """

    def __init__(self, env: Environment, RouterClass, router_id: int, data_series: EventSeries,
                 edges, router_init_args={}, pkg_process_delay: int = 0, **kwargs):
        dyn_env = DynamicEnv(time=lambda: env.now)
        router = RouterClass(dyn_env, router_id=router_id, **router_init_args)
        super().__init__(env, router)
        self.id = router_id
        self.pkg_process_delay = pkg_process_delay
        self.data_series = data_series

        self.interfaces = {v: {"params": ps, "resource": None, "neighbour": None}
                           for (_, v, ps) in edges}
        self.msgProcessingQueue = Resource(self.env, capacity=1)

    def init(self, neighbours: Dict[int, SimpyMessageEnv], config):
        for (v, neighbour) in neighbours.items():
            self.interfaces[v]["neighbour"] = neighbour
            # TODO: initialize edge state properly
            self.interfaces[v]["resource"] = Resource(self.env, capacity=1)

        # Schedule the router initialization
        self.handle(InitMessage(config))

    def _msgEvent(self, msg: Message) -> Event:
        if isinstance(msg, (InitMessage, AddLinkMessage, RemoveLinkMessage)):
            return Event(self.env).succeed(value=msg)

        elif isinstance(msg, OutMessage):
            to = msg.to_node
            return self.env.process(self._edgeTransfer(msg, self.interfaces[to]))

        elif isinstance(msg, InMessage):
            return self.env.process(self._inputQueue(msg))

        elif isinstance(msg, PkgReceivedMessage):
            logger.debug("Package #{} received at node {} at time {}"
                         .format(msg.pkg.id, self.id, self.env.now))
            self.data_series.logEvent(self.env.now, self.env.now - msg.pkg.start_time)

        else:
            raise UnsupportedMessageType(msg)

    def _edgeTransfer(self, msg: OutMessage, edge):
        neighbour = edge["neighbour"]
        inner_msg = msg.inner_msg
        new_msg = InMessage(**msg.getContents())

        # TODO: add an option to enable link clogging with
        # service messages
        if isinstance(inner_msg, ServiceMessage):
            neighbour.handle(new_msg)
            return new_msg

        elif isinstance(inner_msg, PkgMessage):
            pkg = inner_msg.pkg
            logger.debug("Package #{} hop: {} -> {}"
                         .format(pkg.id, msg.from_node, msg.to_node))

            latency = edge["params"]["latency"]
            bandwidth = edge["params"]["bandwidth"]

            with edge["resource"].request() as req:
                yield req
                yield self.env.timeout(pkg.size / bandwidth)

            yield self.env.timeout(latency)
            neighbour.handle(new_msg)
            return new_msg

        else:
            raise UnsupportedMessageType(inner_msg)

    def _inputQueue(self, msg: InMessage):
        inner_msg = msg.inner_msg
        # TODO: add an option for enabling processing delay
        # for service messages
        if isinstance(inner_msg, ServiceMessage):
            return msg

        elif isinstance(inner_msg, PkgMessage):
            with self.msgProcessingQueue.request() as req:
                yield req
                yield self.env.timeout(self.pkg_process_delay)
                return msg

        else:
            raise UnsupportedMessageType(inner_msg)
