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


class NetworkEnvironment(MultiAgentEnv):
    """
    Class which simulates the behavior of computer network
    """
    def __init__(self, RouterClass, data_series: EventSeries,
                 pkg_process_delay: int = 0, default_router_cfg = {}, **kwargs):
        self.RouterClass = RouterClass
        self.default_router_cfg = default_router_cfg
        self.pkg_process_delay = pkg_process_delay
        self.data_series = data_series

        super().__init__(**kwargs)

        self.link_queues = {}
        self.router_queues = {}
        for router_id in self.conn_graph.nodes:
            self.link_queues[router_id] = {}
            for _, nbr in self.conn_graph.edges(router_id):
                self.link_queues[router_id][nbr] = Resource(self.env, capacity=1)
            self.router_queues[router_id] = Resource(self.env, capacity=1)

    def makeConnGraph(self, network_cfg, **kwargs) -> nx.Graph:
        if type(network_cfg) == list:
            return make_network_graph(network_cfg)
        elif type(network_cfg) == dict:
            return gen_network_graph(network_cfg['generator'])
        else:
            raise Exception('Invalid network config: {}'.format(network_cfg))

    def makeHandler(self, agent_id: AgentId) -> MessageHandler:
        assert agent_id[0] == 'router', "Only routers are allowed in computer network"

        dyn_env = DynamicEnv(time=lambda: self.env.now)
        neighbours = [v for _, v in self.conn_graph.edges(agent_id)]

        G = self.conn_graph.to_directed()
        kwargs = make_router_cfg(G, agent_id)
        kwargs.update(self.default_router_cfg)
        if issubclass(self.RouterClass, LinkStateRouter):
            kwargs['adj_links'] = G.adj[agent_id]

        return self.RouterClass(env=dyn_env, id=agent_id, neighbours=neighbours,
                                edge_weight='latency', **kwargs)

    def handleAction(self, from_agent: AgentId, action: Action) -> Event:
        if isinstance(action, PkgRouteAction):
            to_agent = action.to
            if not self.conn_graph.has_edge(from_agent, to_agent):
                raise Exception("Trying to route to a non-neighbor")

            return self.env.process(self._edgeTransfer(from_agent, to_agent, action.pkg))

        elif isinstance(action, PkgReceiveAction):
            logger.debug("Package #{} received at node {} at time {}"
                         .format(action.pkg.id, from_agent[1], self.env.now))

            self.data_series.logEvent(self.env.now, self.env.now - action.pkg.start_time)
            return Event(self.env).succeed()

        else:
            return super().handleAction(from_agent, action)

    def handleWorldEvent(self, event: WorldEvent) -> Event:
        if isinstance(event, PkgEnqueuedEvent):
            self.passToAgent(event.recipient, event)
            return self.env.process(self._inputQueue(event.sender, event.recipient, event.pkg))
        elif isinstance(event, LinkUpdateEvent):
            u_ev = self.passToAgent(event.u, event)
            v_ev = self.passToAgent(event.v, event)
            return u_ev & v_ev
        else:
            return super().handleWorldEvent(event)

    def _edgeTransfer(self, from_agent: AgentId, to_agent: AgentId, pkg: Package):
        logger.debug("Package #{} hop: {} -> {}"
                     .format(pkg.id, from_agent[1], to_agent[1]))

        edge_params = self.conn_graph[from_agent][to_agent]
        latency = edge_params['latency']
        bandwidth = edge_params['bandwidth']

        with self.link_queues[from_agent][to_agent].request() as req:
            yield req
            yield self.env.timeout(pkg.size / bandwidth)

        yield self.env.timeout(latency)
        self.handleWorldEvent(PkgEnqueuedEvent(from_agent, to_agent, pkg))

    def _inputQueue(self, from_agent: AgentId, to_agent: AgentId, pkg: Package):
        with self.router_queues[to_agent].request() as req:
            yield req
            yield self.env.timeout(self.pkg_process_delay)
        self.passToAgent(to_agent, PkgProcessingEvent(from_agent, to_agent, pkg))

class NetworkRunner(SimulationRunner):
    """
    Class which constructs and runs scenarios in computer network simulation
    environment.
    """
    def __init__(self, router_type: str, data_dir=LOG_DATA_DIR+'/network', **kwargs):
        self.router_type = router_type
        super().__init__(data_dir=data_dir, **kwargs)

    def makeMultiAgentEnv(self) -> MultiAgentEnv:
        ChosenRouter = get_router_class(self.router_type, 'network')
        router_cfg = self.run_params['settings']['router'].get(self.router_type, {})

        return NetworkEnvironment(env=self.env, RouterClass=ChosenRouter,
                                  data_series=self.data_series,
                                  network_cfg=self.run_params['network'],
                                  default_router_cfg=router_cfg,
                                  **self.run_params['settings']['router_env'])

    def relevantConfig(self):
        ps = self.run_params
        ss = ps['settings']
        return (ps['network'], ss['pkg_distr'], ss['router_env'],
                ss['router'].get(self.router_type, {}))

    def makeRunId(self, random_seed):
        return '{}-{}'.format(self.router_type, random_seed)

    def runProcess(self, random_seed = None):
        if random_seed is not None:
            set_random_seed(random_seed)

        all_nodes = list(self.world.conn_graph.nodes)
        for node in all_nodes:
            self.world.passToAgent(node, WireInMsg(-1, InitMessage({})))

        pkg_distr = self.run_params['settings']['pkg_distr']

        pkg_id = 1
        for period in pkg_distr["sequence"]:
            try:
                action = period["action"]
                pause = period.get("pause", 0)
                u = ('router', period["u"])
                v = ('router', period["v"])

                if action == 'break_link':
                    self.world.handleWorldEvent(RemoveLinkEvent(u, v))
                elif action == 'restore_link':
                    self.world.handleWorldEvent(AddLinkEvent(u, v, params=self.G.edges[u, v]))
                yield self.env.timeout(pause)

            except KeyError:
                delta = period["delta"]
                try:
                    sources = [('router', v) for v in period["sources"]]
                except KeyError:
                    sources = all_nodes

                try:
                    dests = [('router', v) for v in period["dests"]]
                except KeyError:
                    dests = all_nodes

                simult_sources = period.get("simult_sources", 1)

                for i in range(0, period["pkg_number"] // simult_sources):
                    srcs = random.sample(sources, simult_sources)
                    for src in srcs:
                        dst = random.choice(dests)
                        pkg = Package(pkg_id, DEF_PKG_SIZE, dst, self.env.now, None) # create empty packet
                        logger.debug("Sending random pkg #{} from {} to {} at time {}"
                                     .format(pkg_id, src, dst, self.env.now))
                        self.world.handleWorldEvent(PkgEnqueuedEvent(('world', 0), src, pkg))
                        pkg_id += 1
                    yield self.env.timeout(delta)
