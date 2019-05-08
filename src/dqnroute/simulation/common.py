import networkx as nx
import os

from typing import List
from simpy import Environment, Event, Interrupt
from ..event_series import EventSeries
from ..messages import *
from ..agents import *
from ..utils import data_digest

class MultiAgentEnv:
    """
    Abstract class which simulates an environment with multiple agents,
    where agents are connected accordingly to a given connection graph.
    """
    def __init__(self, env: Environment, **kwargs):
        self.env = env
        self.conn_graph = self.makeConnGraph(**kwargs)

        agent_ids = list(self.conn_graph.nodes)
        self.handlers = {agent_id: self.makeHandler(agent_id) for agent_id in agent_ids}
        self.delayed_msgs = {agent_id: {} for agent_id in agent_ids}

    def makeConnGraph(self, **kwargs) -> nx.Graph:
        """
        A method which defines a connection graph for the system with
        given params.
        Should be overridden. The node labels of a resulting graph should be
        `AgentId`s.
        """
        raise NotImplementedError()

    def makeHandler(self, agent_id: AgentId) -> MessageHandler:
        """
        A method which initializes the agent handler given the agent ID.
        Should be overridden.
        """
        raise NotImplementedError()

    def handle(self, from_agent: AgentId, event: WorldEvent) -> Event:
        """
        Main method which governs how events cause each other in the
        environment. Not to be overridden in children: `handleAction` and
        `handleWorldEvent` should be overridden instead.
        """

        if isinstance(event, Message):
            return self.handleMessage(from_agent, event)
        elif isinstance(event, Action):
            return self.handleAction(from_agent, event)
        elif from_agent[0] == 'world':
            return handleWorldEvent(event)
        else:
            raise Exception('Non-world event: ' + str(event))

    def handleMessage(self, from_agent: AgentId, msg: Message) -> Event:
        """
        Method which handles how messages should be dealt with. Is not meant to be
        overridden.
        """

        if isinstance(msg, DelayedMessage):
            proc = self.env.process(self._delayedHandleGen(from_agent, msg))
            self.delayed_msgs[from_agent][msg.id] = proc
            return proc

        elif isinstance(msg, DelayInterruptMessage):
            try:
                self.delayed_msgs[from_agent][msg.delay_id].interrupt()
            except (KeyError, RuntimeError):
                pass
            return Event(self.env).succeed()

        elif isinstance(msg, WireOutMsg):
            return self.env.process(self._handleOutMsgGen(from_agent, msg))

        else:
            raise UnsupportedMessageType(msg)

    def handleAction(self, from_agent: AgentId, action: Action) -> Event:
        """
        Method which governs how agents' actions influence the environment
        Should be overridden by subclasses.
        """
        raise UnsupportedActionType(action)

    def handleWorldEvent(self, event: WorldEvent) -> Event:
        """
        Method which governs how events from outside influence the environment.
        Should be overridden by subclasses.
        """
        raise UnsupportedEventType(event)

    def passToAgent(self, agent: AgentId, event: WorldEvent) -> Event:
        """
        Let an agent react on event and handle all events produced by agent as
        a consequence.
        """

        for new_event in self.handlers[agent].handle(event):
            self.handle(agent, new_event)
        return Event(self.env).succeed()

    def _delayedHandleGen(self, from_agent: AgentId, msg: DelayedMessage):
        proc_id = msg.id
        delay = msg.delay
        inner = msg.inner_msg

        try:
            yield self.env.timeout(delay)
            yield from self._handleOutMsgGen(from_agent, inner)
        except Interrupt:
            pass
        del self.delayed_msgs[from_agent][proc_id]

    def _handleOutMsgGen(self, from_agent: AgentId, msg: WireOutMsg):
        int_id = msg.interface
        inner = msg.payload

        if int_id == -1:
            to_agent = from_agent
            to_interface = -1
        else:
            to_agent = list(self.conn_graph.edges(from_agent))[int_id][1]
            to_interface = list(self.conn_graph.edges(to_agent)).index((to_agent, from_agent))

        yield self.passToAgent(to_agent, WireInMsg(to_interface, inner))


class SimulationRunner:
    """
    Class which constructs an environment from given settings and runs it.
    """

    def __init__(self, run_params, data_series: EventSeries, data_dir: str, **kwargs):
        self.run_params = run_params
        self.data_series = data_series
        self.data_dir = data_dir

        # Makes a world simulation
        self.env = Environment()
        self.world = self.makeMultiAgentEnv(**kwargs)

    def runDataPath(self, random_seed) -> str:
        cfg = self.relevantConfig()
        return '{}/{}-{}.csv'.format(self.data_dir, data_digest(cfg), self.makeRunId(random_seed))

    def run(self, random_seed = None, ignore_saved = False,
            progress_step = None, progress_queue = None) -> EventSeries:
        """
        Runs the environment, optionally reporting the progress to a given queue
        """
        data_path = self.runDataPath(random_seed)
        run_id = self.makeRunId(random_seed)

        if not ignore_saved and os.path.isfile(data_path):
            self.data_series.load(data_path)
            if progress_queue is not None:
                progress_queue.put((run_id, None))

        else:
            self.env.process(self.runProcess(random_seed))

            if progress_queue is not None:
                if progress_step is None:
                    self.env.run()
                    progress_queue.put((run_id, progress_step))
                else:
                    next_step = progress_step
                    while self.env.peek() != float('inf'):
                        self.env.run(until=next_step)
                        progress_queue.put((run_id, progress_step))
                        next_step += progress_step
                    progress_queue.put((run_id, None))
            else:
                self.env.run()

            os.makedirs(os.path.dirname(data_path), exist_ok=True)
            self.data_series.save(data_path)

        return self.data_series

    def makeMultiAgentEnv(self, **kwargs) -> MultiAgentEnv:
        """
        Initializes a world environment.
        """
        raise NotImplementedError()

    def relevantConfig(self):
        """
        Defines a part of `run_params` which is used to calculate
        run hash (for data saving).
        """
        raise NotImplementedError()

    def makeRunId(self, random_seed):
        """
        Run identificator, which depends on random seed and some run params.
        """
        raise NotImplementedError()

    def runProcess(self, random_seed):
        """
        Generator which generates a series of test scenario events in
        the world environment.
        """
        raise NotImplementedError()


def make_router_cfg(G, RouterClass, router_id, default_cfg={}):
    """
    Helper which makes valid config for the router controller of a given class
    """
    out_routers = [v for (_, v) in G.out_edges(router_id)]
    in_routers = [v for (v, _) in G.in_edges(router_id)]
    router_cfg = {
        'nodes': sorted(list(G.nodes())),
        'edges_num': len(G.edges()), # small hack to make link-state initialization simpler
        'out_neighbours': out_routers,
        'in_neighbours': in_routers
    }
    router_cfg.update(default_cfg)

    if issubclass(RouterClass, LinkStateRouter):
        router_cfg['adj_links'] = G.adj[router_id]
    return router_cfg
