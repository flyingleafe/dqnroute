import networkx as nx
import os

from typing import List
from simpy import Environment, Event, Interrupt
from ..event_series import EventSeries
from ..messages import *
from ..agents import *
from ..utils import data_digest

class SimpyMessageEnv:
    """
    Abstract class which represent an object which processes messages/entities
    from the environment
    """

    def __init__(self, env: Environment, handler: MessageHandler):
        self.env = env
        self.handler = handler
        self.delayed_msgs = {}

    def handle(self, msg: Message) -> Event:
        return self.env.process(self._handleGen(msg))

    def _handleGen(self, msg: Message):
        yield self._msgEvent(msg)
        return [self._msgEvent(m) for m in self.handler.handle(msg)]

    def _delayedHandle(self, proc_id: int, msg: Message, delay: float):
        try:
            yield self.env.timeout(delay)
            self._msgEvent(msg)
        except Interrupt:
            pass
        del self.delayed_msgs[proc_id]

    def _msgEvent(self, msg: Message) -> Event:
        if isinstance(msg, DelayedMessage):
            proc = self.env.process(self._delayedHandle(msg.id, msg.inner_msg, msg.delay))
            self.delayed_msgs[msg.id] = proc
            return Event(self.env).succeed()
        elif isinstance(msg, DelayInterruptMessage):
            try:
                self.delayed_msgs[msg.delay_id].interrupt()
            except (KeyError, RuntimeError):
                pass

            return Event(self.env).succeed()
        else:
            raise UnsupportedMessageType(msg)

class SimulationEnvironment:
    """
    Class which constructs an environment from given settings and runs it.
    """

    def __init__(self, run_params, router_type: str, data_series: EventSeries,
                 data_dir: str, **kwargs):
        self.run_params = run_params
        self.router_type = router_type
        self.data_series = data_series
        self.data_dir = data_dir
        self.context = None   # should be 'network' or 'conveyors' in descendants

        self.G = self.makeGraph(run_params)
        self.env = Environment()

    def makeRouterCfg(self, router_id):
        """
        Makes valid config for the router controller of a given class
        """
        RouterClass = get_router_class(self.router_type, self.context)

        out_routers = [v for (_, v) in self.G.out_edges(router_id)]
        in_routers = [v for (v, _) in self.G.in_edges(router_id)]
        router_cfg = {
            'nodes': sorted(list(self.G.nodes())),
            'out_neighbours': out_routers,
            'in_neighbours': in_routers
        }
        router_cfg.update(self.run_params['settings']['router'].get(self.router_type, {}))

        if issubclass(RouterClass, LinkStateRouter):
            router_cfg['adj_links'] = self.G.adj[router_id]
        return router_cfg

    def runDataPath(self, random_seed) -> str:
        cfg = self.relevantConfig()
        return '{}/{}-{}-{}.csv'.format(self.data_dir, data_digest(cfg), self.router_type, random_seed)

    def run(self, random_seed = None, ignore_saved = False,
            progress_step = None, progress_queue = None) -> EventSeries:
        """
        Runs the environment, optionally reporting the progress to a given queue
        """
        data_path = self.runDataPath(random_seed)
        if not ignore_saved and os.path.isfile(data_path):
            self.data_series.load(data_path)
            if progress_queue is not None:
                progress_queue.put((self.router_type, random_seed, None))

        else:
            self.env.process(self.runProcess(random_seed))

            if progress_queue is not None:
                if progress_step is None:
                    self.env.run()
                    progress_queue.put((self.router_type, random_seed, progress_step))
                else:
                    next_step = progress_step
                    while self.env.peek() != float('inf'):
                        self.env.run(until=next_step)
                        progress_queue.put((self.router_type, random_seed, progress_step))
                        next_step += progress_step
                    progress_queue.put((self.router_type, random_seed, None))
            else:
                self.env.run()

            os.makedirs(os.path.dirname(data_path), exist_ok=True)
            self.data_series.save(data_path)

        return self.data_series

    def makeGraph(self, run_params) -> nx.DiGraph:
        raise NotImplementedError()

    def relevantConfig(self):
        raise NotImplementedError()

    def runProcess(self, random_seed):
        raise NotImplementedError()
