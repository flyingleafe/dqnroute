import logging
import math
import networkx as nx

from typing import List, Callable, Dict, Tuple
from functools import reduce
from simpy import Environment, Event, Resource, Process, Interrupt
from ..utils import *
from ..messages import *
from ..event_series import *
from ..agents import *
from ..constants import *
from .common import *

logger = logging.getLogger(DQNROUTE_LOGGER)

class MultiHandler(MessageHandler):
    """
    Handler which distributes the messages across individual
    sections' handlers and conveyor handler
    """

    def __init__(self, conveyor: MessageHandler, sections: Dict[int, MessageHandler]):
        self._conveyor = conveyor
        self._sections = sections

    def handle(self, msg: Message) -> List[Message]:
        if isinstance(msg, InMessage):
            return self._sections[msg.to_node].handle(msg)

        elif isinstance(msg, InitMessage):
            return sum(map(lambda h: h.handle(msg), self._sections.values()), [])

        elif isinstance(msg, InConveyorMsg):
            msgs = self._conveyor.handle(msg.inner)
            return msgs

        else:
            return super().handle(msg)

class EnergySpender(object):
    """
    Class which records energy consumption
    """
    def __init__(self, env: Environment, data_series: EventSeries, consumption: float):
        self.env = env
        self.data = data_series
        self.consumption = consumption
        self.time_started = -1
        self.total_spent = 0

    def start(self):
        if self.time_started == -1:
            self.time_started = self.env.now

    def stop(self):
        if self.time_started != -1:
            self.data.logUniformRange(self.time_started, self.env.now,
                                      self.consumption)
            self.time_started = -1

class SimpyConveyorEnv(SimpyMessageEnv):
    """
    Conveyor environment which plays the role of outer world and
    handles message passing between routers.
    Passing a message to a router is done via `handle` method.
    """

    def __init__(self, env: Environment, ConveyorClass, RouterClass, conveyor_id: int, sections,
                 time_series: EventSeries, energy_series: EventSeries,
                 speed: float = 1, energy_consumption: float = 1, sec_process_time: int = 10,
                 common_brain: bool = False, conveyor_init_args={}, routers_init_args={}, **kwargs):

        time_func = lambda: env.now
        energy_func = lambda: energy_consumption
        dyn_env = DynamicEnv(time=time_func, energy_consumption=energy_func)

        conveyor = ConveyorClass(env=dyn_env, conveyor_id=conveyor_id, **conveyor_init_args)

        if common_brain and issubclass(RouterClass, DQNRouter):
            common_router_args = next(iter(routers_init_args.values()))
            try:
                embedding_dim = common_router_args['embedding']['dim']
            except KeyError:
                embedding_dim = None
            n = len(common_router_args['nodes'])

            brain = QNetwork(n, embedding_dim=embedding_dim, **common_router_args)
            brain.restore()
            logger.info('Restored model ' + brain._label)
        else:
            brain = None

        section_routers = {
            sec_id: RouterClass(env=dyn_env, router_id=sec_id, brain=brain,
                                **routers_init_args.get(sec_id, {}))
            for sec_id in sections.keys()
        }

        super().__init__(env, MultiHandler(conveyor, section_routers))

        self.id = conveyor_id
        self.time_series = time_series
        self.speed = speed
        self.sec_process_time = sec_process_time
        self.energy_spender = EnergySpender(env, energy_series, energy_consumption)

        self.sections = {}
        for (sec_id, section) in sections.items():
            self.sections[sec_id] = {'resource': Resource(self.env, capacity=1),
                                     'length': section['length']}

        self.last_starting_time = -1

    def init(self, sections_map: Dict[int, SimpyMessageEnv], config):
        self.sections_map = sections_map
        self.handle(InitMessage(config))

    def _msgEvent(self, msg: Message) -> Event:
        if isinstance(msg, InitMessage):
            return Event(self.env).succeed(value=msg)

        elif isinstance(msg, OutMessage):
            new_msg = InMessage(**msg.getContents())
            to_env = self.sections_map[msg.to_node]

            # If we are moving a bag out of our conveyor, notify conveyor
            # controller about it
            if isinstance(msg.inner_msg, PkgMessage) and to_env.id != self.id:
                self.handle(InConveyorMsg(OutgoingBagMsg(msg.inner_msg.pkg)))

            # All outgoing messages are transferred to next section immediately
            to_env.handle(new_msg)

            return Event(self.env).succeed(value=new_msg)

        elif isinstance(msg, InMessage):
            inner = msg.inner_msg

            if isinstance(inner, ServiceMessage):
                return Event(self.env).succeed(value=msg)

            elif isinstance(inner, PkgMessage):
                return self.env.process(self._runSection(msg.from_node, msg.to_node, inner.pkg))

            else:
                raise UnsupportedMessageType(inner)

        elif isinstance(msg, OutConveyorMsg):
            new_msg = InConveyorMsg(msg.inner)
            self.handle(new_msg)
            return Event(self.env).succeed(value=msg)

        elif isinstance(msg, InConveyorMsg):
            inner = msg.inner
            if isinstance(inner, ConveyorStartMsg) and self._idle():
                self.last_starting_time = self.env.now
                self.energy_spender.start()
            elif isinstance(inner, ConveyorStopMsg):
                self.last_starting_time = -1
                self.energy_spender.stop()

            return Event(self.env).succeed(value=msg)

        elif isinstance(msg, PkgReceivedMessage):
            logger.debug("Package #{} received at node {} at time {}"
                         .format(msg.pkg.id, self.id, self.env.now))
            self.time_series.logEvent(self.env.now, self.env.now - msg.pkg.start_time)
            # Notify conveyor that a package has exited
            self.handle(InConveyorMsg(OutgoingBagMsg(msg.pkg)))

        else:
            return super()._msgEvent(msg)

    def _runSection(self, from_sec: int, sec_id: int, bag: Bag):
        """
        Generator which models the process of a bag going along conveyor section
        """
        section = self.sections[sec_id]

        with section['resource'].request() as req:
            yield req
            yield self.env.timeout(self.sec_process_time)

        from_conveyor = -1 if from_sec == -1 else self.sections_map[from_sec].id
        if from_conveyor != self.id:
            self.handle(InConveyorMsg(IncomingBagMsg(bag)))

        yield self.env.timeout(section['length'] / self.speed)

    def _idle(self):
        return self.last_starting_time == -1

class ConveyorsEnvironment(SimulationEnvironment):
    """
    Class which constructs and runs scenarios in conveyor network
    simulation environment.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.context = 'conveyors'
        self.sections_map = {}
        self.conveyors = []
        ChosenRouter = get_router_class(self.router_type, self.context)

        layout = self.run_params['configuration']['layout']
        for (i, conveyor) in enumerate(layout):
            routers_args = {}
            for sec_id in conveyor.keys():
                args = self.makeRouterCfg(sec_id)
                args['edge_weight'] = 'length'
                routers_args[sec_id] = args

            conveyor_args = {'routers': list(conveyor.keys())}
            conveyor_args.update(self.run_params['settings']['conveyor'])

            conveyor_env = SimpyConveyorEnv(self.env, SimpleConveyor, ChosenRouter, i, conveyor,
                                            time_series=self.data_series.subSeries('time'),
                                            energy_series=self.data_series.subSeries('energy'),
                                            conveyor_init_args=conveyor_args,
                                            routers_init_args=routers_args,
                                            **self.run_params['settings']['conveyor_env'])

            for sec_id in conveyor.keys():
                self.sections_map[sec_id] = conveyor_env
            self.conveyors.append(conveyor_env)

        for conveyor in self.conveyors:
            conveyor.init(self.sections_map, {})

    def makeGraph(self, run_params):
        return make_conveyor_graph(run_params['configuration']['layout'])

    def runProcess(self, random_seed = None):
        if random_seed is not None:
            set_random_seed(random_seed)

        bag_distr = self.run_params['settings']['bags_distr']
        sources = self.run_params['configuration']['sources']
        sinks = self.run_params['configuration']['sinks']

        bag_id = 1
        for period in bag_distr['sequence']:
            delta = period['delta']
            cur_sources = period.get('sources', sources)
            cur_sinks = period.get('sinks', sinks)

            for i in range(0, period['bags_number']):
                src = random.choice(cur_sources)
                dst = random.choice(cur_sinks)
                bag = Bag(bag_id, dst, env.now, None)
                logger.debug("Sending random bag #{} from {} to {} at time {}"
                             .format(bag_id, src, dst, self.env.now))
                self.sections_map[src].handle(InMessage(-1, src, PkgMessage(bag)))
                bag_id += 1
                yield self.env.timeout(delta)
