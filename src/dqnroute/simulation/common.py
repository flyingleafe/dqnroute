import networkx as nx
import os
import yaml
import pprint

from typing import List, Optional
from simpy import Environment, Event, Interrupt
from ..event_series import EventSeries
from ..messages import *
from ..agents import *
from ..utils import *


class UnknownAgentError(Exception):
    pass

class HandlerFactory:
    def __init__(self, env: Environment, conn_graph: nx.Graph, **kwargs):
        super().__init__()
        self.env = env
        self.conn_graph = conn_graph
        if self.centralized():
            self.master_handler = self.makeMasterHandler()

    def _makeHandler(self, agent_id: AgentId, **kwargs) -> MessageHandler:
        if self.centralized():
            return SlaveHandler(id=agent_id, master=self.master_handler)
        else:
            neighbours = [v for _, v in self.conn_graph.edges(agent_id)]
            return self.makeHandler(agent_id, neighbours=neighbours, **kwargs)

    def makeMasterHandler(self) -> MasterHandler:
        raise NotImplementedError()

    def makeHandler(self, agent_id: AgentId) -> MessageHandler:
        raise NotImplementedError()

    def handlerClass(self, handler_type: str):
        raise NotImplementedError()

    def centralized(self):
        raise NotImplementedError()


class MultiAgentEnv(HasLog):
    """
    Abstract class which simulates an environment with multiple agents,
    where agents are connected accordingly to a given connection graph.
    """
    def __init__(self, env: Environment, **kwargs):
        self.env = env
        self.conn_graph = self.makeConnGraph(**kwargs)
        self.factory = self.makeHandlerFactory(
            env=self.env, conn_graph=self.conn_graph, **kwargs)

        agent_ids = list(self.conn_graph.nodes)
        self.handlers = {agent_id: self.factory._makeHandler(agent_id) for agent_id in agent_ids}
        self.delayed_evs = {agent_id: {} for agent_id in agent_ids}

        if self.factory.centralized():
            self.handlers[('master', 0)] = self.factory.master_handler
            self.delayed_evs[('master', 0)] = {}

        self._agent_passes = 0

    def time(self):
        return self.env.now

    def logName(self):
        return 'World'

    def makeConnGraph(self, **kwargs) -> nx.Graph:
        """
        A method which defines a connection graph for the system with
        given params.
        Should be overridden. The node labels of a resulting graph should be
        `AgentId`s.
        """
        raise NotImplementedError()

    def makeHandlerFactory(self, **kwargs) -> HandlerFactory:
        """
        Makes a handler factory
        """
        raise NotImplementedError()

    def handle(self, from_agent: AgentId, event: WorldEvent) -> Event:
        """
        Main method which governs how events cause each other in the
        environment. Not to be overridden in children: `handleAction` and
        `handleWorldEvent` should be overridden instead.
        """
        if isinstance(event, MasterEvent):
            from_agent = event.agent
            event = event.inner

        if isinstance(event, Message):
            return self.handleMessage(from_agent, event)

        elif isinstance(event, Action):
            return self.handleAction(from_agent, event)

        elif isinstance(event, DelayedEvent):
            proc = self.env.process(self._delayedHandleGen(from_agent, event))
            self.delayed_evs[from_agent][event.id] = proc
            return Event(self.env).succeed()

        elif isinstance(event, DelayInterrupt):
            try:
                self.delayed_evs[from_agent][event.delay_id].interrupt()
            except (KeyError, RuntimeError):
                pass
            return Event(self.env).succeed()

        elif from_agent[0] == 'world':
            return handleWorldEvent(event)

        else:
            raise Exception('Non-world event: ' + str(event))

    def handleMessage(self, from_agent: AgentId, msg: Message) -> Event:
        """
        Method which handles how messages should be dealt with. Is not meant to be
        overridden.
        """
        if isinstance(msg, WireOutMsg):
            # Out message is considered to be handled as soon as its
            # handling by the recipient is scheduled. We do not
            # wait for other agent to handle them.
            self.env.process(self._handleOutMsgGen(from_agent, msg))
            return Event(self.env).succeed()
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
        if isinstance(event, LinkUpdateEvent):
            return self.handleConnGraphChange(event)
        else:
            raise UnsupportedEventType(event)

    def passToAgent(self, agent: AgentId, event: WorldEvent) -> Event:
        """
        Let an agent react on event and handle all events produced by agent as
        a consequence.
        """
        self._agent_passes += 1
        if agent in self.handlers:
            agent_evs = delayed_first(self.handlers[agent].handle(event))
        elif self.factory.centralized() and isinstance(self.factory.master_handler, Oracle):
            agent_evs = delayed_first(self.factory.master_handler.handleSlaveEvent(node, event))
        else:
            raise UnknownAgentError('No such agent: {}'.format(agent))

        evs = []
        for new_event in agent_evs:
            evs.append(self.handle(agent, new_event))
        return self.env.all_of(evs)

    def handleConnGraphChange(self, event: LinkUpdateEvent) -> Event:
        """
        Adds or removes the connection link and notifies the agents that
        the corresponding intefaces changed availability.
        Connection graph itself does not change to preserve interfaces numbering.
        """
        u = event.u
        v = event.v
        u_int = interface_idx(self.conn_graph, u, v)
        v_int = interface_idx(self.conn_graph, v, u)

        if isinstance(event, AddLinkEvent):
            u_ev = InterfaceSetupEvent(u_int, v, event.params)
            v_ev = InterfaceSetupEvent(v_int, u, event.params)
        elif isinstance(event, RemoveLinkEvent):
            u_ev = InterfaceShutdownEvent(u_int)
            v_ev = InterfaceShutdownEvent(v_int)
        return self.passToAgent(u, u_ev) & self.passToAgent(v, v_ev)

    def _delayedHandleGen(self, from_agent: AgentId, event: DelayedEvent):
        proc_id = event.id
        delay = event.delay
        inner = event.inner

        try:
            yield self.env.timeout(delay)
            self.handle(from_agent, inner)
        except Interrupt:
            pass
        del self.delayed_evs[from_agent][proc_id]

    def _handleOutMsgGen(self, from_agent: AgentId, msg: WireOutMsg):
        int_id = msg.interface
        inner = msg.payload
        to_agent, to_interface = resolve_interface(self.conn_graph, from_agent, int_id)
        yield self.passToAgent(to_agent, WireInMsg(to_interface, inner))


class SimulationRunner:
    """
    Class which constructs an environment from given settings and runs it.
    """

    def __init__(self, run_params, data_dir: str, params_override = {},
                 data_series: Optional[EventSeries] = None, series_period: int = 500,
                 series_funcs: List[str] = ['count', 'sum', 'min', 'max'], **kwargs):
        if type(run_params) == str:
            with open(run_params) as f:
                run_params = yaml.safe_load(f)
        run_params = dict_merge(run_params, params_override)

        if data_series is None:
            data_series = self.makeDataSeries(series_period, series_funcs)

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
            progress_step = None, progress_queue = None, **kwargs) -> EventSeries:
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

    def makeDataSeries(self, series_period, series_funcs):
        """
        Makes a data series if one is not given directly
        """
        raise NotImplementedError()

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

##
# Small run utilities
#

def run_simulation(RunnerClass, return_runner=False, **kwargs):
    runner = RunnerClass(**kwargs)
    data_series = runner.run(**kwargs)
    df = data_series.getSeries(add_avg=True)

    if return_runner:
        return df, runner
    return df

def mk_job_id(router_type, seed):
    return '{}-{}'.format(router_type, seed)

def un_job_id(job_id):
    [router_type, s_seed] = job_id.split('-')
    return router_type, int(s_seed)

def add_cols(df, **cols):
    for (col, val) in cols.items():
        df.loc[:, col] = val

class DummyProgressbarQueue:
    def __init__(self, bar):
        self.bar = bar

    def put(self, val):
        _, delta = val
        if delta is not None:
            self.bar.update(delta)
