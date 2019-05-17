import logging
import pprint
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
from .network import RouterFactory

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


class ConveyorFactory(HandlerFactory):
    def __init__(self, router_type, run_settings, layout, **kwargs):
        super().__init__(**kwargs)
        self.router_type = router_type
        self.layout = layout
        self.conveyor_cfg = run_settings['conveyor']
        self.energy_consumption = run_settings['conveyor_env']['energy_consumption']
        self.max_speed = run_settings['conveyor_env']['speed']

        router_cfg = {'conv_stop_delay': self.conveyor_cfg['stop_delay']}
        router_cfg.update(run_settings['router'].get(router_type, {}))

        self.sub_factory = RouterFactory(router_type, router_cfg, context='conveyors', **kwargs)
        self.centralized = self.sub_factory.centralized

    def makeMasterHandler(self):
        return self.sub_factory.makeMasterHandler()

    def makeHandler(self, agent_id: AgentId, neighbours: List[AgentId]) -> MessageHandler:
        a_type = agent_type(agent_id)
        conv_idx = conveyor_idx(self.env.topology, agent_id)

        if conv_idx != -1:
            dyn_env = self.conveyor_dyn_envs[conv_idx]
        else:
            # only if it's sink
            dyn_env = self.env.copy()

        common_args = {
            'env': dyn_env,
            'id': agent_id,
            'neighbours': neighbours,
            'topology': self.env.topology,
            'router_factory': self.sub_factory
        }

        if a_type == 'conveyor':
            return SimpleRouterConveyor(max_speed=self.max_speed,
                                        length=self.layout['conveyors'][conv_idx]['length'],
                                        **common_args,
                                        **self.conveyor_cfg)
        elif a_type == 'source':
            return RouterSource(**common_args)
        elif a_type == 'sink':
            return RouterSink(**common_args)
        elif a_type == 'diverter':
            return RouterDiverter(**common_args)
        else:
            raise Exception('Unknown agent type: ' + a_type)

    def ready(self):
        self.conveyor_dyn_envs = {}
        for conv_id in self.layout['conveyors'].keys():
            dyn_env = self.env.copy()
            dyn_env.register_var('scheduled_stop', 0)
            self.conveyor_dyn_envs[conv_id] = dyn_env


class ConveyorsEnvironment(MultiAgentEnv):
    """
    Environment which models the conveyor system and the movement of bags.

    TODO: now bags are represented as dots, they should be represented as areas
    """

    def __init__(self, env: Environment, conveyors_layout,
                 time_series: EventSeries, energy_series: EventSeries,
                 speed: float = 1, energy_consumption: float = 1,
                 default_conveyor_args = {}, default_router_args = {}, **kwargs):
        self.max_speed = speed
        self.energy_consumption = energy_consumption
        self.layout = conveyors_layout
        self.time_series = time_series
        self.energy_series = energy_series

        self.topology_graph = make_conveyor_topology_graph(conveyors_layout)

        super().__init__(env=env, conveyors_layout=conveyors_layout, **kwargs)

        # Initialize conveyor-wise state dictionaries
        conv_ids = list(self.layout['conveyors'].keys())

        self.conveyor_models = {}
        for conv_id in conv_ids:
            checkpoints = conveyor_adj_nodes(self.topology_graph, conv_id,
                                             only_own=True, data='conveyor_pos')
            # print('conveyor {} checkpoints: {}, {}'.format(conv_id, checkpoints,
                                                           # self.topology_graph.nodes[checkpoints[-1][0]]))
            length = self.layout['conveyors'][conv_id]['length']
            model = ConveyorModel(length, self.max_speed, checkpoints,
                                  model_id=('world_conv', conv_id))
            self.conveyor_models[conv_id] = model

        self.conveyor_move_proc = {}
        self.conveyor_broken = {conv_id: False for conv_id in conv_ids}

        self.conveyor_energy = {
            conv_id: EnergySpender(self.env, self.energy_series, self.energy_consumption)
            for conv_id in conv_ids
        }

        self.conveyor_upstreams = {
            conv_id: conveyor_adj_nodes(self.topology_graph, conv_id)[-1]
            for conv_id in conv_ids
        }

    def toDynEnv(self) -> DynamicEnv:
        env = super().toDynEnv()
        env.register('energy_consumption', lambda: self.energy_consumption)
        env.register('topology', self.topology_graph)
        return env

    def makeConnGraph(self, conveyors_layout, **kwargs) -> nx.Graph:
        return make_conveyor_conn_graph(conveyors_layout)

    def handleAction(self, from_agent: AgentId, action: Action) -> Event:
        if isinstance(action, BagReceiveAction):
            assert agent_type(from_agent) == 'sink', "Only sink can receive bags!"
            self.log("bag #{} received at sink {}"
                     .format(action.bag.id, from_agent[1]))

            if from_agent != action.bag.dst:
                raise Exception('Bag #{} came to {}, but its destination was {}'
                                .format(action.bag.id, from_agent, action.bag.dst))

            self.time_series.logEvent(self.env.now, self.env.now - action.bag.start_time)
            return Event(self.env).succeed()

        elif isinstance(action, DiverterKickAction):
            assert agent_type(from_agent) == 'diverter', "Only diverter can do kick actions!"
            return self.env.process(self._diverterKick(from_agent))

        elif isinstance(action, ConveyorSpeedChangeAction):
            assert agent_type(from_agent) == 'conveyor', "Only conveyor can change speed!"
            return self.env.process(self._changeConvSpeed(agent_idx(from_agent), action.new_speed))

        else:
            return super().handleAction(from_agent, action)

    def handleWorldEvent(self, event: WorldEvent) -> Event:
        if isinstance(event, BagAppearanceEvent):
            src = ('source', event.src_id)
            bag = event.bag
            self.log("bag #{} appeared at source {}".format(bag.id, src[1]))

            conv_idx, pos = self._nodePos(src)
            ev1 = self.passToAgent(src, BagDetectionEvent(bag))
            return ev1 & self.env.process(self._putBagOnConveyor(conv_idx, bag, pos))

        else:
            return super().handleWorldEvent(event)

    def _diverterKick(self, dv_id: AgentId):
        """
        Checks if some bag is in front of a given diverter now,
        if so, moves this bag from current conveyor to upstream one.
        """
        assert agent_type(dv_id) == 'diverter', "only diverter can kick!!"

        dv_idx = agent_idx(dv_id)
        dv_cfg = self.layout['diverters'][dv_idx]
        conv_idx = dv_cfg['conveyor']
        up_conv = dv_cfg['upstream_conv']
        pos = dv_cfg['pos']

        yield from self._interruptMovement(conv_idx)

        conv_model = self.conveyor_models[conv_idx]
        n_bag, n_pos = conv_model.nearestObject(pos)

        if pos == n_pos:
            conv_model.removeObject(n_bag.id)
            yield from self._putBagOnConveyor(up_conv, n_bag, 0)

        yield from self._resolveAndResume(conv_idx)

    def _changeConvSpeed(self, conv_idx: int, new_speed: float):
        """
        Changes the conveyor speed, updating all the bag movement processes
        accordingly. If the speed just became non-zero, then the conveyor became working;
        if it just became zero, then the conveyor stopped.
        """
        model = self.conveyor_models[conv_idx]
        old_speed = model.speed
        if new_speed == old_speed:
            return

        yield from self._interruptMovement(conv_idx)
        model.setSpeed(new_speed)

        if old_speed == 0:
            self.log('conv {} started!'.format(conv_idx))
            self.conveyor_energy[conv_idx].start()

        if new_speed == 0:
            self.log('conv {} stopped!'.format(conv_idx))
            self.conveyor_energy[conv_idx].stop()

        yield from self._resolveAndResume(conv_idx)

    def _putBagOnConveyor(self, conv_idx, bag, pos):
        """
        Puts a bag on a given position to a given conveyor. If there is currently
        some other bag on a conveyor, throws a `CollisionException`
        """
        yield from self._interruptMovement(conv_idx)

        self.log('bag {} -> conv {} ({}m)'.format(bag.id, conv_idx, pos))
        model = self.conveyor_models[conv_idx]
        model.putObject(bag.id, bag, pos)
        bag.last_conveyor = conv_idx

        yield from self._resolveAndResume(conv_idx)

    def _leaveConveyorEnd(self, conv_idx, bag_id):
        model = self.conveyor_models[conv_idx]
        bag = model.removeObject(bag_id)
        up_node = self.conveyor_upstreams[conv_idx]
        up_type = agent_type(up_node)

        if up_type == 'sink':
            yield self.passToAgent(up_node, BagDetectionEvent(bag))
        elif up_type == 'junction':
            up_conv, up_pos = self._nodePos(up_node)
            yield from self._putBagOnConveyor(up_conv, bag, up_pos)
        else:
            raise Exception('Invalid conveyor upstream node type: ' + up_type)

    def _interruptMovement(self, conv_idx):
        try:
            interrupt = self.conveyor_move_proc[conv_idx].interrupt()
            if interrupt is not None:
                yield interrupt
        except (KeyError, RuntimeError):
            pass
        yield Event(self.env).succeed()

    def _resolveAndResume(self, conv_idx):
        model = self.conveyor_models[conv_idx]

        if model.dirty():
            model.startResolving()
            events = model.immediateEvents()
            self.log('processing events on conv {}: {}'.format(conv_idx, events))

            for bag, node, delay in events:
                atype = agent_type(node)
                if atype == 'junction':
                    # do nothing, we process collisions in another branch
                    pass
                elif atype == 'conv_end':
                    yield from self._leaveConveyorEnd(conv_idx, bag.id)
                elif atype == 'diverter':
                    yield self.passToAgent(node, BagDetectionEvent(bag))
                else:
                    raise Exception('Impossible conv node: {}'.format(node))
            model.endResolving()

        if not model.resolving():
            if conv_idx not in self.conveyor_move_proc:
                move_proc = self.env.process(self._convMovement(conv_idx))
                self.conveyor_move_proc[conv_idx] = move_proc
            else:
                raise Exception('Conveyor move process is still there!')

        yield Event(self.env).succeed()

    def _convMovement(self, conv_idx):
        model = self.conveyor_models[conv_idx]

        model.resume(self.env.now)
        events = model.nextEvents()

        self.log('conveyor {} RESUMED: objs {}, events {}'
                 .format(conv_idx, model.object_positions, events))

        restart = False
        if len(events) > 0:
            try:
                _, _, delay = events[0]
                yield self.env.timeout(delay)
                restart = True
            except Interrupt:
                self.log('conveyor {} INTERRUPT'.format(conv_idx))

        model.pause(self.env.now)
        self.log('conveyor {} PAUSED: objs {}'.format(conv_idx, model.object_positions))
        del self.conveyor_move_proc[conv_idx]
        if restart:
            self.env.process(self._resolveAndResume(conv_idx))

    def _nodePos(self, node: AgentId) -> Tuple[int, int]:
        conv_idx = self.topology_graph.nodes[node]['conveyor']
        pos = self.topology_graph.nodes[node]['conveyor_pos']
        return conv_idx, pos

class ConveyorsRunner(SimulationRunner):
    """
    Class which constructs and runs scenarios in conveyor network
    simulation environment.
    """

    def __init__(self, data_dir=LOG_DATA_DIR+'/conveyors', **kwargs):
        super().__init__(data_dir=data_dir, **kwargs)

    def makeHandlerFactory(self, router_type: str, **kwargs):
        run_settings = self.run_params['settings']
        layout = self.run_params['configuration']
        return ConveyorFactory(router_type, run_settings, layout, **kwargs)

    def makeMultiAgentEnv(self, **kwargs) -> MultiAgentEnv:
        return ConveyorsEnvironment(env=self.env, factory=self.factory,
                                    time_series=self.data_series.subSeries('time'),
                                    energy_series=self.data_series.subSeries('energy'),
                                    conveyors_layout=self.run_params['configuration'],
                                    **self.run_params['settings']['conveyor_env'])

    def relevantConfig(self):
        ps = self.run_params
        ss = ps['settings']
        return (ps['configuration'], ss['bags_distr'], ss['conveyor_env'],
                ss['conveyor'], ss['router'].get(self.factory.router_type, {}))

    def makeRunId(self, random_seed):
        return '{}-{}'.format(self.factory.router_type, random_seed)

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
            # adding a tiny noise to delta
            delta = period['delta'] + round(np.random.normal(0, 0.5), 2)

            cur_sources = period.get('sources', sources)
            cur_sinks = period.get('sinks', sinks)
            simult_sources = period.get("simult_sources", 1)

            for i in range(0, period['bags_number'] // simult_sources):
                srcs = random.sample(cur_sources, simult_sources)
                for src in srcs:
                    dst = random.choice(cur_sinks)
                    bag = Bag(bag_id, ('sink', dst), self.env.now, None)
                    logger.debug("Sending random bag #{} from {} to {} at time {}"
                                 .format(bag_id, src, dst, self.env.now))
                    self.world.handleWorldEvent(BagAppearanceEvent(src, bag))
                    bag_id += 1
                yield self.env.timeout(delta)
