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


class NetworkEnvironment(SimulationEnvironment):
    """
    Class which constructs and runs scenarios in computer network simulation
    environment.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.context = 'network'
        self.routers = {}
        ChosenRouter = get_router_class(self.router_type, self.context)

        for node, nbrs in self.G.adjacency():
            edges = [(node, v, attrs) for v, attrs in nbrs.items()]
            router_args = self.makeRouterCfg(node)
            router_args['edge_weight'] = 'latency'

            self.routers[node] = SimpyRouterEnv(self.env, ChosenRouter,
                                                router_id=node,
                                                router_init_args=router_args,
                                                data_series=self.data_series, edges=edges,
                                                **self.run_params['settings']['router_env'])

        for node in self.G.nodes():
            out_routers = {v: self.routers[v] for (_, v) in self.G.out_edges(node)}
            self.routers[node].init(out_routers, {})

    def makeGraph(self, run_params):
        return make_network_graph(run_params['network'])

    def runProcess(self, random_seed = None):
        if random_seed is not None:
            set_random_seed(random_seed)

        pkg_distr = self.run_params['settings']['pkg_distr']

        pkg_id = 1
        for period in pkg_distr["sequence"]:
            try:
                action = period["action"]
                pause = period["pause"]
                u = period["u"]
                v = period["v"]

                if action == 'break_link':
                    self.routers[u].handle(RemoveLinkMessage(v))
                    self.routers[v].handle(RemoveLinkMessage(u))
                elif action == 'restore_link':
                    self.routers[u].handle(AddLinkMessage(v, params=self.G.edges[u, v]))
                    self.routers[v].handle(AddLinkMessage(u, params=self.G.edges[v, u]))
                yield self.env.timeout(pause)

            except KeyError:
                delta = period["delta"]
                all_nodes = list(self.routers.keys())
                sources = period.get("sources", all_nodes)
                dests = period.get("dests", all_nodes)

                for i in range(0, period["pkg_number"]):
                    src = random.choice(sources)
                    dst = random.choice(dests)
                    pkg = Package(pkg_id, 1024, dst, self.env.now, None) # create empty packet
                    logger.debug("Sending random pkg #{} from {} to {} at time {}"
                                 .format(pkg_id, src, dst, self.env.now))
                    self.routers[src].handle(InMessage(-1, src, PkgMessage(pkg)))
                    pkg_id += 1
                    yield self.env.timeout(delta)
