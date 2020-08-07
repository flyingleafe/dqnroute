import logging
import pprint
import math
import numpy as np
import networkx as nx
import os

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
REAL_BAG_RADIUS = 0.5

class ConveyorFactory(HandlerFactory):
    def __init__(self, router_type, routers_cfg, topology,
                 conn_graph, conveyor_cfg, energy_consumption, max_speed,
                 conveyor_models, oracle=True, **kwargs):
        self.router_type = router_type
        self.router_cfg = routers_cfg.get(router_type, {})
        self.conveyor_cfg = conveyor_cfg
        self.topology = topology
        self.conveyor_cfg = conveyor_cfg
        self.energy_consumption = energy_consumption
        self.max_speed = max_speed

        self.conveyor_models = conveyor_models
        self.oracle = oracle

        stop_delay = self.conveyor_cfg['stop_delay']
        try:
            routers_cfg[router_type]['conv_stop_delay'] = stop_delay
        except KeyError:
            routers_cfg[router_type] = {'conv_stop_delay': stop_delay}

        self.RouterClass = get_router_class(router_type, 'conveyors', oracle=oracle)

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

        for conv_id in self.conveyor_models.keys():
            dyn_env = self.dynEnv()
            dyn_env.register_var('prev_total_nrg', 0)
            dyn_env.register_var('total_nrg', 0)
            self.conveyor_dyn_envs[conv_id] = dyn_env

    def centralized(self):
        return issubclass(self.RouterClass, MasterHandler)

    def dynEnv(self):
        time_func = lambda: self.env.now
        energy_func = lambda: self.energy_consumption
        return DynamicEnv(time=time_func, energy_consumption=energy_func)

    def makeMasterHandler(self) -> MasterHandler:
        dyn_env = self.dynEnv()
        cfg = {**self.router_cfg, **self.conveyor_cfg}
        if self.oracle:
            cfg['conveyor_models'] = self.conveyor_models
        else:
            conv_lengths = {cid: model.length
                            for (cid, model) in self.conveyor_models.items()}
            cfg['conv_lengths'] = conv_lengths

        return self.RouterClass(env=dyn_env, topology=self.topology,
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
            'router_factory': self.sub_factory,
            'oracle': self.oracle
        }

        if a_type == 'conveyor':
            if self.oracle:
                common_args['model'] = self.conveyor_models[conv_idx]
                common_args['all_models'] = self.conveyor_models

            return SimpleRouterConveyor(max_speed=self.max_speed,
                                        length=self.conveyor_models[conv_idx].length,
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

    def __init__(self, env: Environment, conveyors_layout, data_series: EventSeries,
                 speed: float = 1, energy_consumption: float = 1,
                 default_conveyor_args = {}, default_router_args = {}, **kwargs):
        self.max_speed = speed
        self.energy_consumption = energy_consumption
        self.layout = conveyors_layout
        self.data_series = data_series

        self.topology_graph = make_conveyor_topology_graph(conveyors_layout)

        # Initialize conveyor-wise state dictionaries
        conv_ids = list(self.layout['conveyors'].keys())
        dyn_env = DynamicEnv(time=lambda: self.env.now)

        self.conveyor_models = {}
        for conv_id in conv_ids:
            checkpoints = conveyor_adj_nodes(self.topology_graph, conv_id,
                                             only_own=True, data='conveyor_pos')
            length = self.layout['conveyors'][conv_id]['length']
            model = ConveyorModel(dyn_env, length, self.max_speed,
                                  checkpoints, self.data_series.subSeries('energy'),
                                  model_id=('world_conv', conv_id))
            self.conveyor_models[conv_id] = model

        self.conveyors_move_proc = None
        self.current_bags = {}
        self.conveyor_broken = {conv_id: False for conv_id in conv_ids}

        self.conveyor_upstreams = {
            conv_id: conveyor_adj_nodes(self.topology_graph, conv_id)[-1]
            for conv_id in conv_ids
        }

        super().__init__(
            env=env, topology=self.topology_graph, conveyors_layout=conveyors_layout,
            energy_consumption=energy_consumption, max_speed=speed, **kwargs)

        self._updateAll()

    def log(self, msg, force=False):
        if force:
            super().log(msg, True)

    def makeHandlerFactory(self, **kwargs):
        kwargs['conveyor_models'] = self.conveyor_models
        return ConveyorFactory(**kwargs)

    def makeConnGraph(self, conveyors_layout, **kwargs) -> nx.Graph:
        return make_conveyor_conn_graph(conveyors_layout)

    def handleAction(self, from_agent: AgentId, action: Action) -> Event:
        if isinstance(action, BagReceiveAction):
            assert agent_type(from_agent) == 'sink', "Only sink can receive bags!"
            bag = action.bag

            self.log("bag #{} received at sink {}"
                     .format(bag.id, from_agent[1]))

            if from_agent != bag.dst:
                raise Exception('Bag #{} came to {}, but its destination was {}'
                                .format(action.bag.id, from_agent, bag.dst))

            assert bag.id in self.current_bags, "why leave twice??"
            self.current_bags.pop(action.bag.id)

            self.data_series.logEvent('time', self.env.now, self.env.now - action.bag.start_time)
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

            conv_idx = conveyor_idx(self.topology_graph, src)
            self.passToAgent(src, BagDetectionEvent(bag))
            return self._checkInterrupt(lambda: self._putBagOnConveyor(conv_idx, src, bag, src))

        elif isinstance(event, ConveyorBreakEvent):
            return self._checkInterrupt(lambda: self._conveyorBreak(event.conv_idx))
        elif isinstance(event, ConveyorRestoreEvent):
            return self._checkInterrupt(lambda: self._conveyorRestore(event.conv_idx))
        else:
            return super().handleWorldEvent(event)

    def _checkInterrupt(self, callback):
        if self.conveyors_move_proc is None:
            callback()
        else:
            try:
                self.conveyors_move_proc.interrupt()
                self.conveyors_move_proc = None
            except RuntimeError as err:
                self.log('UNEXPECTED INTERRUPT FAIL {}'.format(err), True)

            for model in self.conveyor_models.values():
                model.pause()

            callback()
            self._updateAll()

        return Event(self.env).succeed()

    def _conveyorBreak(self, conv_idx: int):
        self.log('conv break: {}'.format(conv_idx), True)
        model = self.conveyor_models[conv_idx]
        model.setSpeed(0)
        self.log('chill bags: {}'.format(len(model.objects)), True)

        self.conveyor_broken[conv_idx] = True
        for aid in self.handlers.keys():
            self.passToAgent(aid, ConveyorBreakEvent(conv_idx))

    def _conveyorRestore(self, conv_idx: int):
        self.log('conv restore: {}'.format(conv_idx), True)
        self.conveyor_broken[conv_idx] = False
        for aid in self.handlers.keys():
            self.passToAgent(aid, ConveyorRestoreEvent(conv_idx))

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
            self._removeBagFromConveyor(conv_idx, n_bag.id, dv_id)
            self._putBagOnConveyor(up_conv, dv_id, n_bag, dv_id)

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
        if new_speed == 0:
            self.log('conv {} stopped!'.format(conv_idx))

    def _removeBagFromConveyor(self, conv_idx, bag_id, node):
        model = self.conveyor_models[conv_idx]
        bag = model.removeObject(bag_id)
        conv_aid = ('conveyor', conv_idx)
        self.passToAgent(conv_aid, OutgoingBagEvent(bag, node))
        return bag

    def _putBagOnConveyor(self, conv_idx, sender, bag, node):
        """
        Puts a bag on a given position to a given conveyor. If there is currently
        some other bag on a conveyor, throws a `CollisionException`
        """
        if self.conveyor_broken[conv_idx]:
            # just forget about the bag and say we had a collision
            self.data_series.logEvent('collisions', self.env.now, 1)
            self.current_bags.pop(bag.id)
            return

        pos = node_conv_pos(self.topology_graph, conv_idx, node)
        assert pos is not None, "adasdasdasdas!"

        self.log('bag {} -> conv {} ({}m)'.format(bag.id, conv_idx, pos))
        model = self.conveyor_models[conv_idx]
        nearest = model.putObject(bag.id, bag, pos, return_nearest=True)
        if nearest is not None:
            n_oid, n_pos = nearest
            if abs(pos - n_pos) < 2*REAL_BAG_RADIUS:
                self.log('collision detected: (#{}; {}m) with (#{}; {}m) on conv {}'
                         .format(bag.id, pos, n_oid, n_pos, conv_idx), True)
                self.data_series.logEvent('collisions', self.env.now, 1)

        bag.last_conveyor = conv_idx
        conv_aid = ('conveyor', conv_idx)
        self.current_bags[bag.id].add(node)
        self.passToAgent(conv_aid, IncomingBagEvent(sender, bag, node))

    def _leaveConveyorEnd(self, conv_idx, bag_id) -> bool:
        bag = self._removeBagFromConveyor(conv_idx, bag_id, ('conv_end', conv_idx))
        up_node = self.conveyor_upstreams[conv_idx]
        up_type = agent_type(up_node)

        if up_type == 'sink':
            self.passToAgent(up_node, BagDetectionEvent(bag))
            return True

        if up_type == 'junction':
            up_conv = conveyor_idx(self.topology_graph, up_node)
            self._putBagOnConveyor(up_conv, ('conveyor', conv_idx), bag, up_node)
        else:
            raise Exception('Invalid conveyor upstream node type: ' + up_type)
        return False

    def _updateAll(self):
        self.log('CHO PO')
        self.conveyors_move_proc = None

        left_to_sinks = set()
        # Resolving all immediate events
        for (conv_idx, (bag, node, delay)) in all_unresolved_events(self.conveyor_models):
            assert delay == 0, "well that's just obnoxious"
            if self.conveyor_broken[conv_idx]:
                continue

            if bag.id in left_to_sinks or node in self.current_bags[bag.id]:
                continue

            self.log('conv {}: handling {} on {}'.format(conv_idx, bag, node))

            model = self.conveyor_models[conv_idx]
            atype = agent_type(node)
            left_to_sink = False

            if atype == 'junction':
                self.passToAgent(('conveyor', conv_idx), PassedBagEvent(bag, node))
            elif atype == 'conv_end':
                left_to_sink = self._leaveConveyorEnd(conv_idx, bag.id)
                if left_to_sink:
                    left_to_sinks.add(bag.id)
            elif atype == 'diverter':
                self.passToAgent(node, BagDetectionEvent(bag))
                if bag.id in model.objects:
                    self.passToAgent(('conveyor', conv_idx), PassedBagEvent(bag, node))
            else:
                raise Exception('Impossible conv node: {}'.format(node))

            if bag.id in self.current_bags and bag.id not in left_to_sinks:
                self.current_bags[bag.id].add(node)

        for conv_idx, model in self.conveyor_models.items():
            if model.resolving():
                model.endResolving()
            model.resume()

        self.conveyors_move_proc = self.env.process(self._move())

    def _move(self):
        try:
            events = all_next_events(self.conveyor_models)
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
                model.pause()

            self._updateAll()
        except Interrupt:
            pass


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
        collisions_series = event_series(series_period, series_funcs)
        return MultiEventSeries(time=time_series, energy=energy_series, collisions=collisions_series)

    def makeMultiAgentEnv(self, **kwargs) -> MultiAgentEnv:
        run_settings = self.run_params['settings']
        return ConveyorsEnvironment(env=self.env, data_series=self.data_series,
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
            if self.world.factory.centralized():
                seed = random_seed + 42
            else:
                seed = random_seed
            set_random_seed(seed)

        all_nodes = list(self.world.conn_graph.nodes)
        for node in all_nodes:
            self.world.passToAgent(node, WireInMsg(-1, InitMessage({})))

        bag_distr = self.run_params['settings']['bags_distr']
        sources = list(self.run_params['configuration']['sources'].keys())
        sinks = self.run_params['configuration']['sinks']

        # Little pause in order to let all initialization messages settle
        yield self.env.timeout(1)

        bag_id = 1
        
        # added by Igor to support loading already trained models
        if "IGOR_OMIT_TRAINING" in os.environ:
            return
        
        for period in bag_distr['sequence']:
            try:
                action = period['action']
                conv_idx = period['conv_idx']
                pause = period.get('pause', 0)

                if pause > 0:
                    yield self.env.timeout(pause)

                if action == 'conv_break':
                    yield self.world.handleWorldEvent(ConveyorBreakEvent(conv_idx))
                elif action == 'conv_restore':
                    yield self.world.handleWorldEvent(ConveyorRestoreEvent(conv_idx))
                else:
                    raise Exception('Unknown action: ' + action)

                if pause > 0:
                    yield self.env.timeout(pause)

            except KeyError:
                # adding a tiny noise to delta
                delta = period['delta'] + round(np.random.normal(0, 0.5), 2)

                cur_sources = period.get('sources', sources)
                cur_sinks = period.get('sinks', sinks)
                simult_sources = period.get("simult_sources", 1)
                #print(period, cur_sources)
                #assert False

                for i in range(0, period['bags_number'] // simult_sources):
                    srcs = random.sample(cur_sources, simult_sources)
                    for (j, src) in enumerate(srcs):
                        if j > 0:
                            mini_delta = round(abs(np.random.normal(0, 0.5)), 2)
                            yield self.env.timeout(mini_delta)

                        dst = random.choice(cur_sinks)
                        bag = Bag(bag_id, ('sink', dst), self.env.now, None)
                        logger.debug(f"Sending random bag #{bag_id} from {src} to {dst} at time {self.env.now}")
                        yield self.world.handleWorldEvent(BagAppearanceEvent(src, bag))

                        bag_id += 1
                    yield self.env.timeout(delta)
