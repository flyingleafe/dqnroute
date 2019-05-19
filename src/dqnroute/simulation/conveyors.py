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

DIVERTER_RANGE = 0.5

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
    def __init__(self, router_type, routers_cfg, topology, conn_graph, conveyors_layout,
                 conveyor_cfg, energy_consumption, max_speed, **kwargs):
        self.router_type = router_type
        self.router_cfg = routers_cfg.get(router_type, {})
        self.conveyor_cfg = conveyor_cfg
        self.topology = topology
        self.layout = conveyors_layout
        self.conveyor_cfg = conveyor_cfg
        self.energy_consumption = energy_consumption
        self.max_speed = max_speed

        stop_delay = self.conveyor_cfg['stop_delay']
        try:
            routers_cfg[router_type]['conv_stop_delay'] = stop_delay
        except KeyError:
            routers_cfg[router_type] = {'conv_stop_delay': stop_delay}

        self.RouterClass = get_router_class(router_type, 'conveyors')

        if not self.centralized():
            r_topology, _, _ = conv_to_router(topology)
            self.sub_factory = RouterFactory(
                router_type, routers_cfg,
                conn_graph=r_topology.to_undirected(),
                topology_graph=r_topology,
                context='conveyors', **kwargs)

        super().__init__(conn_graph=conn_graph, **kwargs)

        self.conveyor_dyn_envs = {}
        time_func = lambda: self.env.now
        energy_func = lambda: self.energy_consumption

        for conv_id in self.layout['conveyors'].keys():
            dyn_env = self.dynEnv()
            dyn_env.register_var('scheduled_stop', 0)
            self.conveyor_dyn_envs[conv_id] = dyn_env

    def centralized(self):
        return issubclass(self.RouterClass, MasterHandler)

    def dynEnv(self):
        time_func = lambda: self.env.now
        energy_func = lambda: self.energy_consumption
        return DynamicEnv(time=time_func, energy_consumption=energy_func)

    def makeMasterHandler(self) -> MasterHandler:
        dyn_env = self.dynEnv()
        conv_lengths = {cid: conv['length']
                        for (cid, conv) in self.layout['conveyors'].items()}
        cfg = {**self.router_cfg, **self.conveyor_cfg}
        return self.RouterClass(env=dyn_env, topology=self.topology,
                                conv_lengths=conv_lengths,
                                max_speed=self.max_speed, **cfg)

    def makeHandler(self, agent_id: AgentId, neighbours: List[AgentId], **kwargs) -> MessageHandler:
        a_type = agent_type(agent_id)
        conv_idx = conveyor_idx(self.topology, agent_id)

        if conv_idx != -1:
            dyn_env = self.conveyor_dyn_envs[conv_idx]
        else:
            # only if it's sink
            dyn_env = self.dynEnv()

        common_args = {
            'env': dyn_env,
            'id': agent_id,
            'neighbours': neighbours,
            'topology': self.topology,
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


class ConveyorsEnvironment(MultiAgentEnv):
    """
    Environment which models the conveyor system and the movement of bags.
    """

    def __init__(self, conveyors_layout, time_series: EventSeries, energy_series: EventSeries,
                 speed: float = 1, energy_consumption: float = 1,
                 default_conveyor_args = {}, default_router_args = {}, **kwargs):
        self.max_speed = speed
        self.energy_consumption = energy_consumption
        self.layout = conveyors_layout
        self.time_series = time_series
        self.energy_series = energy_series

        self.topology_graph = make_conveyor_topology_graph(conveyors_layout)

        super().__init__(
            topology=self.topology_graph, conveyors_layout=conveyors_layout,
            energy_consumption=energy_consumption, max_speed=speed, **kwargs)

        # Initialize conveyor-wise state dictionaries
        conv_ids = list(self.layout['conveyors'].keys())

        self.conveyor_models = {}
        for conv_id in conv_ids:
            checkpoints = conveyor_adj_nodes(self.topology_graph, conv_id,
                                             only_own=True, data='conveyor_pos')
            length = self.layout['conveyors'][conv_id]['length']
            model = ConveyorModel(length, self.max_speed, checkpoints,
                                  model_id=('world_conv', conv_id))
            self.conveyor_models[conv_id] = model

        self.conveyors_move_proc = None
        self.current_bags = {}
        self.conveyor_broken = {conv_id: False for conv_id in conv_ids}

        self.conveyor_energy = {
            conv_id: EnergySpender(self.env, self.energy_series, self.energy_consumption)
            for conv_id in conv_ids
        }

        self.conveyor_upstreams = {
            conv_id: conveyor_adj_nodes(self.topology_graph, conv_id)[-1]
            for conv_id in conv_ids
        }

        self._updateAll()

    def log(self, msg, force=False):
        if force:
            super().log(msg, True)

    def makeHandlerFactory(self, **kwargs):
        return ConveyorFactory(**kwargs)

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

            self.current_bags.pop(action.bag.id)
            self.time_series.logEvent(self.env.now, self.env.now - action.bag.start_time)
            return Event(self.env).succeed()

        elif isinstance(action, DiverterKickAction):
            assert agent_type(from_agent) == 'diverter', "Only diverter can do kick actions!"
            self.log('diverter {} kicks'.format(agent_idx(from_agent)))

            return self._checkInterrupt(lambda: self._diverterKick(from_agent))

        elif isinstance(action, ConveyorSpeedChangeAction):
            assert agent_type(from_agent) == 'conveyor', "Only conveyor can change speed!"
            self.log('change conv {} speed to {}'
                     .format(agent_idx(from_agent), action.new_speed))

            return self._checkInterrupt(
                lambda: self._changeConvSpeed(agent_idx(from_agent), action.new_speed))

        else:
            return super().handleAction(from_agent, action)

    def handleWorldEvent(self, event: WorldEvent) -> Event:
        if isinstance(event, BagAppearanceEvent):
            src = ('source', event.src_id)
            bag = event.bag
            self.current_bags[bag.id] = set()
            self.log("bag #{} appeared at source {}".format(bag.id, src[1]))

            conv_idx, pos = self._nodePos(src)
            self.passToAgent(src, BagDetectionEvent(bag))
            return self._checkInterrupt(lambda: self._putBagOnConveyor(conv_idx, bag, pos))

        else:
            return super().handleWorldEvent(event)

    def _checkInterrupt(self, callback):
        if self.conveyors_move_proc is None:
            self.log('EBASH SRAZU')
            callback()
        else:
            try:
                self.conveyors_move_proc.interrupt()
                self.conveyors_move_proc = None
            except RuntimeError as err:
                self.log('POSOSANO??? {}'.format(err))

            for model in self.conveyor_models.values():
                model.pause(self.env.now)

            callback()
            self._updateAll()

        return Event(self.env).succeed()

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

        conv_model = self.conveyor_models[conv_idx]
        n_bag, n_pos = conv_model.nearestObject(pos)

        if abs(pos - n_pos) <= DIVERTER_RANGE:
            conv_model.removeObject(n_bag.id)
            self._putBagOnConveyor(up_conv, n_bag, 0)

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

        model.setSpeed(new_speed)

        if old_speed == 0:
            self.log('conv {} started!'.format(conv_idx))
            self.conveyor_energy[conv_idx].start()

        if new_speed == 0:
            self.log('conv {} stopped!'.format(conv_idx))
            self.conveyor_energy[conv_idx].stop()

    def _putBagOnConveyor(self, conv_idx, bag, pos):
        """
        Puts a bag on a given position to a given conveyor. If there is currently
        some other bag on a conveyor, throws a `CollisionException`
        """
        self.log('bag {} -> conv {} ({}m)'.format(bag.id, conv_idx, pos))
        model = self.conveyor_models[conv_idx]
        model.putObject(bag.id, bag, pos)
        bag.last_conveyor = conv_idx

    def _leaveConveyorEnd(self, conv_idx, bag_id):
        model = self.conveyor_models[conv_idx]
        bag = model.removeObject(bag_id)
        up_node = self.conveyor_upstreams[conv_idx]
        up_type = agent_type(up_node)

        if up_type == 'sink':
            self.passToAgent(up_node, BagDetectionEvent(bag))
        elif up_type == 'junction':
            up_conv, up_pos = self._nodePos(up_node)
            self._putBagOnConveyor(up_conv, bag, up_pos)
        else:
            raise Exception('Invalid conveyor upstream node type: ' + up_type)

    def _startResolving(self):
        for model in self.conveyor_models.values():
            if model.dirty():
                model.startResolving()
        self.conveyors_move_proc = None

    def _endResolving(self):
        for model in self.conveyor_models.values():
            if model.resolving():
                model.endResolving()
            model.resume(self.env.now)
        self.conveyors_move_proc = self.env.process(self._move())

    def _updateAll(self):
        self._startResolving()
        self.log('CHO PO')

        # Resolving all immediate events
        for (conv_idx, (bag, node, delay)) in self._allUnresolvedEvents():
            assert delay == 0, "well that's just obnoxious"
            if node in self.current_bags[bag.id]:
                continue

            model = self.conveyor_models[conv_idx]

            self.log('conv {}: handling {} on {}'.format(conv_idx, bag, node))

            atype = agent_type(node)
            if atype == 'junction':
                # do nothing, we process collisions in another branch
                pass
            elif atype == 'conv_end':
                self._leaveConveyorEnd(conv_idx, bag.id)
            elif atype == 'diverter':
                self.passToAgent(node, BagDetectionEvent(bag))
            else:
                raise Exception('Impossible conv node: {}'.format(node))

            if bag.id in self.current_bags:
                self.current_bags[bag.id].add(node)

        self._endResolving()

    def _move(self):
        try:
            events = self._allNextEvents()
            self.log('MOVING: {}'.format(events))

            if len(events) > 0:
                conv_idx, (bag, node, delay) = events[0]
                assert delay > 0, "next event delay is 0!"
                self.log('NEXT EVENT: conv {} - ({}, {}, {})'.format(conv_idx, bag, node, delay))
                yield self.env.timeout(delay)
            else:
                # hang forever (until interrupt)
                yield Event(self.env)

            for model in self.conveyor_models.values():
                model.pause(self.env.now)

            self._updateAll()
        except Interrupt:
            pass

    def _allUnresolvedEvents(self):
        while True:
            had_some = False
            for conv_idx, model in self.conveyor_models.items():
                if model.resolving():
                    ev = model.pickUnresolvedEvent()
                    if ev is not None:
                        yield (conv_idx, ev)
                        had_some = True
            if not had_some:
                break

    def _allNextEvents(self):
        res = []
        for conv_idx, model in self.conveyor_models.items():
            evs = model.nextEvents()
            res = merge_sorted(res, [(conv_idx, ev) for ev in evs],
                               using=lambda p: p[1][2])
        return res

    def _nodePos(self, node: AgentId) -> Tuple[int, int]:
        conv_idx = self.topology_graph.nodes[node]['conveyor']
        pos = self.topology_graph.nodes[node]['conveyor_pos']
        return conv_idx, pos

class ConveyorsRunner(SimulationRunner):
    """
    Class which constructs and runs scenarios in conveyor network
    simulation environment.
    """
    context = 'conveyors'

    def __init__(self, data_dir=LOG_DATA_DIR+'/conveyors', **kwargs):
        super().__init__(data_dir=data_dir, **kwargs)

    def makeDataSeries(self, series_period, series_funcs):
        time_series = event_series(series_period, series_funcs)
        energy_series = event_series(series_period, series_funcs)
        return MultiEventSeries(time=time_series, energy=energy_series)

    def makeMultiAgentEnv(self, **kwargs) -> MultiAgentEnv:
        run_settings = self.run_params['settings']
        return ConveyorsEnvironment(env=self.env, time_series=self.data_series.subSeries('time'),
                                    energy_series=self.data_series.subSeries('energy'),
                                    conveyors_layout=self.run_params['configuration'],
                                    routers_cfg=run_settings['router'],
                                    conveyor_cfg=run_settings['conveyor'],
                                    **run_settings['conveyor_env'], **kwargs)

    def relevantConfig(self):
        ps = self.run_params
        ss = ps['settings']
        return (ps['configuration'], ss['bags_distr'], ss['conveyor_env'],
                ss['conveyor'], ss['router'].get(self.world.factory.router_type, {}))

    def makeRunId(self, random_seed):
        return '{}-{}'.format(self.world.factory.router_type, random_seed)

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
                for (j, src) in enumerate(srcs):
                    if j > 0:
                        mini_delta = round(abs(np.random.normal(0, 0.5)), 2)
                        yield self.env.timeout(mini_delta)

                    dst = random.choice(cur_sinks)
                    bag = Bag(bag_id, ('sink', dst), self.env.now, None)
                    logger.debug("Sending random bag #{} from {} to {} at time {}"
                                 .format(bag_id, src, dst, self.env.now))
                    yield self.world.handleWorldEvent(BagAppearanceEvent(src, bag))

                    bag_id += 1
                yield self.env.timeout(delta)

        # Stop the hanging of move process
        # self.world.conveyors_move_proc.interrupt()
