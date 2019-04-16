import logging
import math

from typing import List, Callable, Dict
from simpy import Environment, Event, Resource, Process
from ..utils import *
from ..messages import *
from ..event_series import *
from ..agents import *
from ..constants import *
from .common import *

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
        router = RouterClass(env=dyn_env, router_id=router_id, **router_init_args)
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
            return super()._msgEvent(msg)

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

#
# Runner functions
#

def _run_scenario(env: Environment, routers: Dict[str, SimpyRouterEnv],
                  G: nx.DiGraph, pkg_distr, random_seed = None):
    if random_seed is not None:
        set_random_seed(random_seed)

    pkg_id = 1
    for period in pkg_distr["sequence"]:
        try:
            action = period["action"]
            pause = period["pause"]
            u = period["u"]
            v = period["v"]

            if action == 'break_link':
                routers[u].handle(RemoveLinkMessage(v))
                routers[v].handle(RemoveLinkMessage(u))
            elif action == 'restore_link':
                routers[u].handle(AddLinkMessage(v, G.edges[u, v]))
                routers[v].handle(AddLinkMessage(u, G.edges[v, u]))
            yield env.timeout(pause)
        except KeyError:
            delta = period["delta"]
            all_nodes = list(routers.keys())
            sources = period.get("sources", all_nodes)
            dests = period.get("dests", all_nodes)

            for i in range(0, period["pkg_number"]):
                src = random.choice(sources)
                dst = random.choice(dests)
                pkg = Package(pkg_id, 1024, dst, env.now, None) # create empty packet
                logger.debug("Sending random pkg #{} from {} to {} at time {}"
                             .format(pkg_id, src, dst, env.now))
                routers[src].handle(InMessage(-1, src, PkgMessage(pkg)))
                pkg_id += 1
                yield env.timeout(delta)

def _get_router_class(router_type):
    if router_type == "simple_q":
        return SimpleQRouterNetwork
    elif router_type == "link_state":
        return LinkStateRouter
    elif router_type == "dqn":
        return DQNRouterNetwork
    else:
        raise Exception("Unsupported router type: " + router_type)

def run_network_scenario(run_params, router_type: str, event_series: EventSeries,
                         random_seed = None, progress_step = None, progress_queue = None) -> EventSeries:
    """
    Runs a computer network test scenario with given params and
    aggregates test data in a given `EventSeries`.
    """

    G = make_network_graph(run_params['network'])

    env = Environment()
    routers = {}
    ChosenRouter = _get_router_class(router_type)

    for node, nbrs in G.adjacency():
        edges = [(node, v, attrs) for v, attrs in nbrs.items()]
        router_args = make_router_cfg(ChosenRouter, node, G, run_params)
        router_args['edge_weight'] = 'latency'

        routers[node] = SimpyRouterEnv(env, ChosenRouter,
                                       router_id=node,
                                       router_init_args=router_args,
                                       data_series=event_series, edges=edges,
                                       **run_params['settings']['router_env'])

    for node in G.nodes():
        out_routers = {v: routers[v] for (_, v) in G.out_edges(node)}
        routers[node].init(out_routers, {})

    logger.setLevel(logging.DEBUG)
    env.process(_run_scenario(env, routers, G, run_params['settings']['pkg_distr'], random_seed))
    run_env_progress(env, router_type=router_type, random_seed=random_seed,
                     progress_step=progress_step, progress_queue=progress_queue)

    return event_series
