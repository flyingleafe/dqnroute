import logging
import math
import numpy as np
import networkx as nx

from typing import List, Callable, Dict, Tuple
from functools import reduce
from simpy import Environment, Event, Resource, Process, Interrupt
from ..utils import *
from ..conveyor_model import *
from ..messages import *
from ..event_series import *
from ..agents import *
from ..constants import *
from .common import *

logger = logging.getLogger(DQNROUTE_LOGGER)


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


class ConveyorsEnvironment(MultiAgentEnv):
    """
    Environment which models the conveyor system and the movement of bags.

    TODO: now bags are represented as dots, they should be represented as areas
    """

    def __init__(self, env: Environment, RouterClass, ConveyorClass, conveyors_layout,
                 time_series: EventSeries, energy_series: EventSeries,
                 speed: float = 1, energy_consumption: float = 1,
                 default_conveyor_args = {}, default_router_args = {}, **kwargs):
        self.RouterClass = RouterClass
        self.ConveyorClass = ConveyorClass
        self.default_router_cfg = default_router_args
        self.default_conveyor_cfg = default_conveyor_args
        self.max_speed = speed
        self.energy_consumption = energy_consumption
        self.layout = conveyors_layout
        self.time_series = time_series
        self.energy_series = energy_series

        self.topology_graph = make_conveyor_topology_graph(conveyors_layout)

        if issubclass(self.RouterClass, MasterHandler):
            dyn_env = DynamicEnv(time=lambda: env.now)
            master_cfg = {**self.default_conveyor_cfg, **self.default_router_cfg}
            self.master_handler = self.RouterClass(env=dyn_env, topology=self.topology_graph,
                                                   layout=self.layout, **master_cfg)

        super().__init__(env=env, conveyors_layout=conveyors_layout, **kwargs)

        # Initialize conveyor-wise state dictionaries
        conv_ids = list(self.layout['conveyors'].keys())
        self.conveyor_bags = {conv_id: {} for conv_id in conv_ids}
        self.conveyor_statuses = {
            conv_id: {'broken': False, 'working': False,
                      'speed': 0, 'length': self.layout['conveyors'][conv_id]['length'],
                      'power': EnergySpender(self.env, self.time_series, self.energy_consumption)}
            for conv_id in conv_ids
        }

        self.conveyor_nodes = {conv_id: [] for conv_id in conv_ids}
        for node, ps in self.topology_graph.nodes(data=True):
            if agent_type(node) in ('diverter', 'junction'):
                conv = ps['conveyor']
                pos = ps['conveyor_pos']
                self.conveyor_nodes[conv].append((node, pos))
        for ls in self.conveyor_nodes.values():
            ls.sort(key=lambda p: p[1])

        self.conveyor_upstreams = {}
        for conv_id in conv_ids:
            up = self.layout['conveyors'][conv_id]['upstream']
            if up['type'] == 'sink':
                self.conveyor_upstreams[conv_id] = ('sink', up['idx'])
            else:
                up_conv = up['idx']
                up_pos = up['pos']
                self.conveyor_upstreams[conv_id] = self._nextStopNode(up_conv, up_pos - 1)

    def makeConnGraph(self, conveyors_layout, **kwargs) -> nx.Graph:
        return make_conveyor_conn_graph(conveyors_layout)

    def makeHandler(self, agent_id: AgentId) -> MessageHandler:
        if issubclass(self.RouterClass, MasterHandler):
            return SlaveHandler(id=agent_id, master=self.master_router)
        else:
            time_func = lambda: self.env.now
            energy_func = lambda: energy_consumption
            dyn_env = DynamicEnv(time=time_func, energy_consumption=energy_func)
            neighbours = [v for (_, v) in self.conn_graph.edges(agent_id)]
            a_type = agent_type(agent_id)

            if a_type == 'conveyor':
                routers = [] # TODO: fill those in
                return SimpleConveyor(env=dyn_env, id=agent_id, neighbours=neighbours, routers=routers,
                                      **self.default_conveyor_cfg)
            elif a_type == 'source':
                return ItemSource(env=dyn_env, id=agent_id, neighbours=neighbours)
            elif a_type == 'sink':
                return ItemSink(env=dyn_env, id=agent_id, neighbours=neighbours)
            elif a_type == 'diverter':
                host_conv_idx = self.topology_graph.nodes[agent_id]['conveyor']
                return RouterDiverter(env=dyn_env, id=agent_id, neighbours=neighbours,
                                      topology_graph=self.topology_graph,
                                      RouterClass=self.RouterClass, host_conveyor=('conveyor', host_conv_idx),
                                      router_args=self.default_router_cfg)
            else:
                raise Exception('Unknown agent type: ' + a_type)

    def handleAction(self, from_agent: AgentId, action: Action) -> Event:
        if isinstance(action, BagReceiveAction):
            assert agent_type(from_agent) == 'sink', "Only sink can receive bags!"
            logger.debug("Bag #{} received at sink {} at time {}"
                         .format(action.bag.id, from_agent[1], self.env.now))

            self.time_series.logEvent(self.env.now, self.env.now - action.bag.start_time)
            return Event(self.env).succeed()

        elif isinstance(action, DiverterKickAction):
            assert agent_type(from_agent) == 'diverter', "Only diverter can do kick actions!"
            return self.env.process(self._diverterKickGen(from_agent))

        elif isinstance(action, ConveyorSpeedChangeAction):
            assert agent_type(from_agent) == 'conveyor', "Only conveyor can change speed!"
            return self.env.process(self._changeConvSpeedGen(agent_idx(from_agent), action.new_speed))

        else:
            return super().handleAction(from_agent, action)

    def handleWorldEvent(self, event: WorldEvent) -> Event:
        if isinstance(event, BagAppearanceEvent):
            src = ('source', event.src_id)
            bag = event.bag
            conv_idx, pos = self._nodePos(src)
            self.passToAgent(src, BagDetectionEvent(bag))
            self.env.process(self._putBagOnConveyorGen(conv_idx, bag, pos))

        else:
            return super().handleWorldEvent(event)

    def _diverterKickGen(self, dv_id: AgentId):
        """
        Checks if some bag is in front of a given diverter now,
        if so, moves this bag from current conveyor to upstream one.
        """

        dv_idx = agent_idx(dv_id)
        dv_cfg = self.layout['diverters'][dv_idx]
        conv_idx = dv_cfg['conveyor']
        up_conv = dv_cfg['upstream_conv']
        pos = dv_cfg['pos']

        yield self._interruptMovement(conv_idx)

        kicked_bag_id = -1
        for (bid, status) in self.conveyor_bags[conv_idx].items():
            if status['pos'] == pos:
                kicked_bag_id = bid
                break

        if kicked_bag_id != -1:
            kicked_bag_status = self.conveyor_bags[conv_idx].pop(kicked_bag_id)
            yield from self._putBagOnConveyorGen(up_conv, kicked_bag_status['bag'], 0)

        yield self._resumeMovement(conv_idx)

    def _changeConvSpeedGen(self, conv_idx: int, new_speed: float):
        """
        Changes the conveyor speed, updating all the bag movement processes
        accordingly. If the speed just became non-zero, then the conveyor became working;
        if it just became zero, then the conveyor stopped.
        """

        assert new_speed >= 0, "Negative speed is not allowed!"
        if new_speed > self.max_speed:
            raise Exception('Conveyor #{} tried to set speed {}, which is more than max {}'
                            .format(conv_idx, new_speed, self.max_speed))

        conv_status = self.conveyor_statuses[conv_idx]
        old_speed = conv_status['speed']
        if new_speed == old_speed:
            return

        if old_speed == 0:
            conv_status['working'] = True
            conv_status['power'].start()
        else:
            yield self._interruptMovement(conv_idx)

        conv_status['speed'] = new_speed

        if new_speed == 0:
            conv_status['working'] = False
            conv_status['power'].stop()
        else:
            yield self._resumeMovement(conv_idx)

    def _putBagOnConveyorGen(self, conv_idx, bag, pos):
        """
        Puts a bag on a given position to a given conveyor. If there is currently
        some other bag on a conveyor, throws a `CollisionException`
        """

        yield self._interruptMovement(conv_idx)
        for (bid, status) in self.conveyor_bags[conv_idx].items():
            if status['pos'] == pos:
                raise CollisionException(bag, status['bag'], pos)

        self.conveyor_bags[conv_idx][bag.id] = {'bag': bag, 'pos': pos}
        yield self._resumeMovement(conv_idx)

    def _bagArrivalGen(self, node: AgentId, bag_id: int):
        """
        It is a generator so that it can yield `_interruptMovement` to
        update bag positions properly
        """

        atype = agent_type(node)
        conv_idx = agent_idx(node) if atype == 'conv_end' else self.topology_graph.nodes[node]['conveyor']
        bag = self.conveyor_bags[conv_idx][bag_id]['bag']

        if atype == 'junction':
            # do nothing, we process collisions in another branch
            pass
        elif atype == 'conv_end':
            yield from self._leaveConveyorEndGen(conv_idx, bag_id)
        elif atype == 'diverter':
            # We do not use `passToAgent` here, because if the diverter kicks, we
            # should wait for changes in bags positions to take effect
            for new_event in self.handlers[node].handle(BagDetectionEvent(bag)):
                if isinstance(new_event, Action):
                    yield self.handle(node, new_event)
                else:
                    self.handle(node, new_event)

        if bag_id in self.conveyor_bags[conv_idx]:
            yield self._resumeMovement(conv_idx, bag_id)

    def _leaveConveyorEndGen(self, conv_idx, bag_id):
        bag_status = self.conveyor_bags[conv_idx].pop(bag_id)
        bag = bag_status['bag']
        up_node = self.conveyor_upstreams[conv_idx]
        up_type = agent_type(up_node)

        if up_type == 'sink':
            yield self.passToAgent(up_node, BagDetectionEvent(bag))
        elif up_type == 'junction':
            up_conv, up_pos = self._nodePos(up_node)
            yield from self._putBagOnConveyorGen(up_conv, bag, up_pos)
        else:
            raise Exception('Invalid conveyor upstream node type: ' + up_type)

    def _interruptMovement(self, conv_idx, bag_ids=None) -> Event:
        if bag_ids is None:
            bag_ids = list(self.conveyor_bags[conv_idx].keys())
        elif type(bag_ids) == int:
            bag_ids = [bag_ids]
        for bid in bag_ids:
            status_dict = self.conveyor_bags[conv_idx][bid]
            try:
                move_proc = status_dict.pop('proc')
                move_proc.interrupt()
            except (KeyError, RuntimeError):
                pass

        # `_interruptMovement` should be yield so that bag positions
        # are updated after movement interruptions
        return Event(self.env).succeed()

    def _resumeMovement(self, conv_idx, bag_ids=None) -> Event:
        if bag_ids is None:
            bag_ids = list(self.conveyor_bags[conv_idx].keys())
        elif type(bag_ids) == int:
            bag_ids = [bag_ids]
        for bid in bag_ids:
            move_proc = self.env.process(self._bagMovementGen(conv_idx, bid))
            self.conveyor_bags[conv_idx][bid]['proc'] = move_proc
        return Event(self.env).succeed()

    def _bagMovementGen(self, conv_idx, bag_id):
        speed = self.conveyor_statuses[conv_idx]['speed']
        start_pos = self.conveyor_bags[conv_idx][bag_id]['pos']
        start_time = self.env.now
        try:
            n_id, n_pos = self._nextStopNode(conv_idx, pos)
            dist = n_pos - start_pos
            yield self.env.timeout(dist / speed)
            self.conveyor_bags[conv_idx][bag_id]['pos'] = n_pos
            self.env.process(self._bagArrivalGen(n_id, bag_id))
        except Interrupt:
            time_passed = self.env.now - start_time
            new_pos = start_pos + (speed * time_passed)
            self.conveyor_bags[conv_idx][bag_id]['pos'] = new_pos

    def _nextStopNode(self, conv_idx, pos):
        for (node, n_pos) in self.conveyor_nodes[conv_idx]:
            if n_pos > pos:
                return node, n_pos
        return ('conv_end', conv_idx), self.conveyor_statuses[conv_idx]['length']

    def _nodePos(self, node: AgentId) -> Tuple[int, int]:
        conv_idx = self.topology_graph.nodes[node]['conveyor']
        pos = self.topology_graph.nodes[node]['conveyor_pos']
        return conv_idx, pos

class ConveyorsRunner(SimulationRunner):
    """
    Class which constructs and runs scenarios in conveyor network
    simulation environment.
    """

    def __init__(self, router_type: str, data_dir=LOG_DATA_DIR+'/conveyors', **kwargs):
        self.router_type = router_type
        super().__init__(data_dir=data_dir, **kwargs)

    def makeMultiAgentEnv(self) -> MultiAgentEnv:
        ChosenRouter = get_router_class(self.router_type, 'conveyors')
        router_cfg = self.run_params['settings']['router'].get(self.router_type, {})
        conveyor_cfg = self.run_params['settings']['conveyor']

        return ConveyorsEnvironment(env=self.env, RouterClass=ChosenRouter,
                                    ConveyorClass=SimpleConveyor,
                                    time_series=self.data_series.subSeries('time'),
                                    energy_series=self.data_series.subSeries('energy'),
                                    conveyors_layout=self.run_params['configuration'],
                                    default_router_args=router_cfg,
                                    default_conveyor_args=conveyor_cfg,
                                    **self.run_params['settings']['conveyor_env'])

    def relevantConfig(self):
        ps = self.run_params
        ss = ps['settings']
        return (ps['configuration'], ss['bags_distr'], ss['conveyor_env'],
                ss['conveyor'], ss['router'].get(self.router_type, {}))

    def makeRunId(self, random_seed):
        return '{}-{}'.format(self.router_type, random_seed)

    def runProcess(self, random_seed = None):
        if random_seed is not None:
            set_random_seed(random_seed)

        all_nodes = list(self.world.conn_graph.nodes)
        for node in all_nodes:
            self.world.passToAgent(node, WireInMsg(-1, InitMessage({})))

        bag_distr = self.run_params['settings']['bags_distr']
        sources = list(self.run_params['configuration']['sources'].keys())
        sinks = self.run_params['configuration']['sinks']

        # Little pause in order to let all initialization messages settle
        yield self.env.timeout(1)

        bag_id = 1
        for period in bag_distr['sequence']:
            delta = period['delta']
            cur_sources = period.get('sources', sources)
            cur_sinks = period.get('sinks', sinks)
            simult_sources = period.get("simult_sources", 1)

            for i in range(0, period['bags_number'] // simult_sources):
                srcs = random.sample(cur_sources, simult_sources)
                for src in srcs:
                    dst = random.choice(cur_sinks)
                    bag = Bag(bag_id, dst, self.env.now, None)
                    logger.debug("Sending random bag #{} from {} to {} at time {}"
                                 .format(bag_id, src, dst, self.env.now))
                    self.world.handleWorldEvent(BagAppearanceEvent(src, bag))
                    bag_id += 1
                yield self.env.timeout(delta)
