import logging
import math

from typing import List, Callable, Dict
from simpy import Environment, Event, Resource, Process, Interrupt
from ..utils import *
from ..messages import *
from ..event_series import EventSeries
from ..agents import Router, MessageHandler
from ..constants import DQNROUTE_LOGGER
from .message_env import SimpyMessageEnv

logger = logging.getLogger(DQNROUTE_LOGGER)

class MultiHandler(MessageHandler):
    """
    Handler which distributes the messages across individual
    sections' handlers
    """

    def __init__(self, sections: Dict[int, MessageHandler]):
        self._sections = sections

    def handle(self, msg: Message) -> List[Message]:
        if isinstance(msg, InMessage):
            return self._sections[msg.to_node].handle(msg)

        elif isinstance(msg, InitMessage):
            return sum(map(lambda h: h.handle(msg), self._sections.values()), [])

        else:
            return super().handle(msg)

class SimpyConveyorEnv(SimpyMessageEnv):
    """
    Conveyor environment which plays the role of outer world and
    handles message passing between routers.
    Passing a message to a router is done via `handle` method.
    """

    def __init__(self, env: Environment, RouterClass, conveyor_id: int, sections,
                 time_series: EventSeries, energy_series: EventSeries,
                 speed: float = 1, energy_consumption: float = 1, stop_delay: float = 50,
                 sec_process_time: int = 10, routers_init_args={}, **kwargs):

        time_func = lambda: env.now
        energy_func = lambda bag_id: self.bag_energy_gap.pop(bag_id)
        dyn_env = DynamicEnv(time=time_func, energy_gap=energy_func)

        section_routers = {
            sec_id: RouterClass(dyn_env, router_id=sec_id,
                                **routers_init_args.get(sec_id, {}))
            for sec_id in sections.keys()
        }

        super().__init__(env, MultiHandler(section_routers))

        self.id = conveyor_id
        self.time_series = time_series
        self.energy_series = energy_series
        self.speed = speed
        self.energy_consumption = energy_consumption
        self.stop_delay = stop_delay
        self.sec_process_time = sec_process_time

        self.sections = {}
        for (sec_id, section) in sections.items():
            self.sections[sec_id] = {'resource': Resource(self.env, capacity=1),
                                     'length': section['length']}

        # On the start, the conveyor is idle
        self.working_process = self.env.process(empty_gen())
        self.stopping_time = 0
        self.bag_energy_gap = {}

    def init(self, sections_map: Dict[int, SimpyMessageEnv], config):
        self.sections_map = sections_map
        self.handle(InitMessage(config))

    def _msgEvent(self, msg: Message) -> Event:
        if isinstance(msg, InitMessage):
            return Event(self.env).succeed(value=msg)

        elif isinstance(msg, OutMessage):
            new_msg = InMessage(**msg.getContents())
            to_env = self.sections_map[msg.to_node]

            # All outgoing messages are transferred to next section immediately
            to_env.handle(new_msg)
            return Event(self.env).succeed(value=new_msg)

        elif isinstance(msg, InMessage):
            inner = msg.inner_msg

            if isinstance(inner, ServiceMessage):
                return Event(self.env).succeed(value=msg)

            elif isinstance(inner, PkgMessage):
                return self.env.process(self._runSection(msg.to_node, inner.pkg))

            else:
                raise UnsupportedMessageType(inner)

        elif isinstance(msg, PkgReceivedMessage):
            logger.debug("Package #{} received at node {} at time {}"
                         .format(msg.pkg.id, self.id, self.env.now))
            self.time_series.logEvent(self.env.now, self.env.now - msg.pkg.start_time)
            self.energy_series.logEvent(self.env.now, msg.pkg.energy_overhead)

        else:
            raise UnsupportedMessageType(msg)

    def _runSection(self, sec_id: int, bag: Bag):
        """
        Generator which models the process of a bag going alogn conveyor section
        """
        section = self.sections[sec_id]

        with section['resource'].request() as req:
            yield req
            yield self.env.timeout(self.sec_process_time)

        working_time_overhead = self.stop_delay - max(0, self.stopping_time - self.env.now)
        energy_overhead = working_time_overhead * self.energy_consumption
        bag.spend_energy(energy_overhead)
        self.bag_energy_gap[bag.id] = energy_overhead
        self._launch_if_idle()

        yield self.env.timeout(section['length'] / self.speed)

    def _launch_if_idle(self):
        if not self._idle():
            self.working_process.interrupt()
        self.working_process = self.env.process(self._workingProcess())

    def _idle(self):
        return not self.working_process.is_alive

    def _workingProcess(self):
        try:
            self.stopping_time = self.env.now + self.stop_delay
            yield self.env.timeout(self.stop_delay)
        except Interrupt:
            pass
