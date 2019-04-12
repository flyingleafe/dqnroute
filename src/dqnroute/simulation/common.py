from typing import List
from simpy import Environment, Event, Interrupt
from ..messages import *
from ..agents import *

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
            if msg.id in self.delayed_msgs:
                raise Exception('okay well what the fuck? {}'.format(msg.id))
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

def make_router_cfg(RouterClass, router_id, G, run_params):
    """
    Makes valid config for the router controller of a given class
    """

    out_routers = [v for (_, v) in G.out_edges(router_id)]
    in_routers = [v for (v, _) in G.in_edges(router_id)]
    router_cfg = {
        'nodes': sorted(list(G.nodes())),
        'out_neighbours': out_routers,
        'in_neighbours': in_routers
    }
    router_cfg.update(run_params['settings']['router'])

    if issubclass(RouterClass, LinkStateRouter):
        router_cfg['adj_links'] = G.adj[router_id]
    return router_cfg

def run_env_progress(env: Environment, router_type: str, random_seed = None,
                     progress_step = None, progress_queue = None):
    """
    Runs the environment, optionally reporting the progress to a given queue
    """

    if progress_queue is not None:
        if progress_step is None:
            env.run()
            progress_queue.put((router_type, random_seed, progress_step))
        else:
            next_step = progress_step
            while env.peek() != float('inf'):
                env.run(until=next_step)
                progress_queue.put((router_type, random_seed, progress_step))
                next_step += progress_step
        progress_queue.put((router_type, random_seed, None))
    else:
        env.run()
