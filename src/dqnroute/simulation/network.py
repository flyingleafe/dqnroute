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
from .training import TrainingRouterClass

logger = logging.getLogger(DQNROUTE_LOGGER)

class RouterFactory(HandlerFactory):
    def __init__(self, router_type, routers_cfg, context = None,
                 topology_graph = None, training_router_type = None, **kwargs):
        RouterClass = get_router_class(router_type, context)
        self.context = context
        self.router_cfg = routers_cfg.get(router_type, {})
        self.edge_weight = 'latency' if context == 'network' else 'length'
        self._dyn_env = None

        if training_router_type is None:
            self.training_mode = False
            self.router_type = router_type
            self.RouterClass = RouterClass
        else:
            self.training_mode = True
            TrainerClass = get_router_class(training_router_type, context)
            self.router_type = 'training__{}__{}'.format(router_type, training_router_type)
            self.RouterClass = TrainingRouterClass(RouterClass, TrainerClass, **kwargs)
        super().__init__(**kwargs)

        if topology_graph is None:
            self.topology_graph = self.conn_graph.to_directed()
        else:
            self.topology_graph = topology_graph

        if self.training_mode:
            dummy = RouterClass(
                **self._handlerArgs(('router', 0), neighbours=[], random_init=True))
            self.brain = dummy.brain
            self.router_cfg['brain'] = self.brain

    def dynEnv(self):
        if self._dyn_env is None:
            return DynamicEnv(time=lambda: self.env.now)
        else:
            return self._dyn_env

    def useDynEnv(self, env):
        self._dyn_env = env

    def makeMasterHanlder(self) -> MasterHandler:
        dyn_env = self.dynEnv()
        return self.RouterClass(
            env=dyn_env, network=self.topology_graph,
            edge_weight='latency', **self.router_cfg)

    def _handlerArgs(self, agent_id, **kwargs):
        G = self.topology_graph
        kwargs.update({
            'env': self.dynEnv(),
            'id': agent_id,
            'edge_weight': self.edge_weight,
            'nodes': sorted(list(G.nodes())),
            'edges_num': len(G.edges()), # small hack to make link-state initialization simpler
        })
        kwargs.update(self.router_cfg)

        if issubclass(self.RouterClass, LinkStateRouter):
            kwargs['adj_links'] = G.adj[agent_id]
        return kwargs

    def makeHandler(self, agent_id: AgentId, **kwargs) -> MessageHandler:
        assert agent_id[0] == 'router', "Only routers are allowed in computer network"
        return self.RouterClass(**self._handlerArgs(agent_id, **kwargs))

    def centralized(self):
        return issubclass(self.RouterClass, MasterHandler)

class NetworkEnvironment(MultiAgentEnv):
    """
    Class which simulates the behavior of computer network
    """
    def __init__(self, data_series: EventSeries, pkg_process_delay: int = 0, **kwargs):
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

    def makeHandlerFactory(self, **kwargs):
        return RouterFactory(context='network', **kwargs)

    def handleAction(self, from_agent: AgentId, action: Action) -> Event:
        if isinstance(action, PkgRouteAction):
            to_agent = action.to
            if not self.conn_graph.has_edge(from_agent, to_agent):
                raise Exception("Trying to route to a non-neighbor")

            self.env.process(self._edgeTransfer(from_agent, to_agent, action.pkg))
            return Event(self.env).succeed()

        elif isinstance(action, PkgReceiveAction):
            logger.debug("Package #{} received at node {} at time {}"
                         .format(action.pkg.id, from_agent[1], self.env.now))

            self.data_series.logEvent(self.env.now, self.env.now - action.pkg.start_time)
            return Event(self.env).succeed()

        else:
            return super().handleAction(from_agent, action)

    def handleWorldEvent(self, event: WorldEvent) -> Event:
        if isinstance(event, PkgEnqueuedEvent):
            self.env.process(self._inputQueue(event.sender, event.recipient, event.pkg))
            return self.passToAgent(event.recipient, event)
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
    context = 'network'

    def __init__(self, data_dir=LOG_DATA_DIR+'/network', **kwargs):
        super().__init__(data_dir=data_dir, **kwargs)

    def makeDataSeries(self, series_period, series_funcs):
        return event_series(series_period, series_funcs)

    def makeMultiAgentEnv(self, **kwargs) -> MultiAgentEnv:
        return NetworkEnvironment(env=self.env, data_series=self.data_series,
                                  network_cfg=self.run_params['network'],
                                  routers_cfg=self.run_params['settings']['router'],
                                  **self.run_params['settings']['router_env'], **kwargs)

    def relevantConfig(self):
        ps = self.run_params
        ss = ps['settings']
        return (ps['network'], ss['pkg_distr'], ss['router_env'],
                ss['router'].get(self.world.factory.router_type, {}))

    def makeRunId(self, random_seed):
        return '{}-{}'.format(self.world.factory.router_type, random_seed)

    def runProcess(self, random_seed = None):
        if random_seed is not None:
            set_random_seed(random_seed)

        all_nodes = list(self.world.conn_graph.nodes)
        all_edges = list(self.world.conn_graph.edges(data=True))
        broken_edges = []
        for node in all_nodes:
            self.world.passToAgent(node, WireInMsg(-1, InitMessage({})))

        pkg_distr = self.run_params['settings']['pkg_distr']

        pkg_id = 1
        for period in pkg_distr["sequence"]:
            try:
                action = period["action"]
                pause = period.get("pause", 0)
                is_random = period.get("random", False)

                if action == 'break_link':
                    if is_random:
                        i = random.randint(0, len(all_edges-1))
                        u, v, ps = all_edges.pop(i)
                        broken_edges.append((u, v, ps))
                    else:
                        u = ('router', period["u"])
                        v = ('router', period["v"])

                    yield self.world.handleWorldEvent(RemoveLinkEvent(u, v))
                elif action == 'restore_link':
                    if is_random:
                        if len(broken_edges) == 0:
                            continue

                        i = random.randint(0, len(broken_edges-1))
                        u, v, ps = broken_edges.pop(i)
                        all_edges.append((u, v, ps))
                    else:
                        u = ('router', period["u"])
                        v = ('router', period["v"])

                    yield self.world.handleWorldEvent(AddLinkEvent(u, v, params=self.world.conn_graph.edges[u, v]))

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
                        yield self.world.handleWorldEvent(PkgEnqueuedEvent(('world', 0), src, pkg))
                        pkg_id += 1
                    yield self.env.timeout(delta)
