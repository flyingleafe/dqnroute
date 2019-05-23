import networkx as nx
import random

from typing import List, Tuple, Dict
from ..base import *
from ...messages import *
from ...utils import *
from ...conveyor_model import *

class CentralizedController(MasterHandler):
    """
    A centralized controller for the whole conveyor system
    """
    def __init__(self, conv_lengths: Dict[int, float], topology: nx.DiGraph,
                 max_speed: float, stop_delay: float, **kwargs):
        super().__init__(**kwargs)
        self.topology = topology
        self.max_speed = max_speed
        self.stop_delay = stop_delay

        for u, v in self.topology.edges:
            self.topology[u][v]['max_allowed_speed'] = self.max_speed

        self.conveyor_models = {}
        for (conv_id, length) in conv_lengths.items():
            checkpoints = conveyor_adj_nodes(self.topology, conv_id,
                                             only_own=True, data='conveyor_pos')
            model = ConveyorModel(self.env, length, self.max_speed, checkpoints,
                                  model_id=('conveyor', conv_id))
            self.conveyor_models[conv_id] = model

        conv_ids = list(conv_lengths.keys())
        self.max_conv_speeds = {cid: self.max_speed for cid in conv_ids}
        self.conv_delayed_stops = {cid: -1 for cid in conv_ids}
        self.delayed_update = -1
        self.current_bags = {}
        self.bag_convs = {}

        self.sink_bags = {}
        for node in self.topology.nodes:
            if agent_type(node) == 'sink':
                self.sink_bags[node] = set()

        self.last_cascade_update_time = -1

        evs = self._update()
        assert len(evs) == 0, "doing something on first update? hey!"

    def log(self, msg, force=False):
        super().log(msg, force)

    def handleSlaveEvent(self, slave_id: AgentId, event: WorldEvent) -> List[WorldEvent]:
        if isinstance(event, (IncomingBagEvent, OutgoingBagEvent)):
            # no oracling here
            return []
        elif isinstance(event, BagDetectionEvent):
            bag = event.bag
            atype = agent_type(slave_id)

            evs = self._interrupt()

            if atype == 'source':
                evs += self.introduceBag(slave_id, bag)
            elif atype == 'diverter':
                evs += self.divertBag(slave_id, bag)
            elif atype == 'sink':
                evs += self.finishBag(slave_id, bag)
            else:
                raise Exception('Bag detected on {} somehow!'.format(slave_id))

            return evs + self._update()
        else:
            return super().handleSlaveEvent(slave_id, event)

    def introduceBag(self, src: AgentId, bag: Bag) -> List[WorldEvent]:
        self.log('NEW bag {} ({})'.format(bag, src))

        self.current_bags[bag.id] = set()
        conv_idx = conveyor_idx(self.topology, src)
        return self.putBagToConv(conv_idx, bag, 0)

    def divertBag(self, dv: AgentId, bag: Bag) -> List[WorldEvent]:
        nxt = self.routeBag(dv, bag)
        cur_conv = conveyor_idx(self.topology, dv)
        next_conv = self.topology[dv][nxt]['conveyor']

        model = self.conveyor_models[cur_conv]
        dv_pos = model.checkpointPos(dv)
        o_pos = model.objPos(bag.id)

        if o_pos != dv_pos:
            if o_pos is None:
                if bag.id in self.bag_convs:
                    false_conv = self.bag_convs[bag.id]
                    false_model = self.conveyor_models[false_conv]
                    false_pos = true_model.objPos(bag.id)
                    self.log('DIVERGENCE WITH REALITY: conv {}: {} passed the {} in {}m, while we thought it is on conv {} in {}m'
                             .format(cur_conv, bag, dv, dv_pos, false_conv, false_pos), True)
                    # false_model.removeObject(bag.id)
                elif bag.id in self.sink_bags[nxt]:
                    self.log('DIVERGENCE WITH REALITY: conv {}: {} passed the {} in {}m, while we thought it has reached the {}'
                             .format(cur_conv, bag, dv, dv_pos, nxt), True)
                    # self.sink_bags[nxt].remove(bag.id)
                else:
                    raise Exception('Okay this is epic!')

                # self.bag_convs[bag.id] = cur_conv
                # model.putObject(bag.id, bag, dv_pos)
            else:
                self.log('DIVERGENCE WITH REALITY: conv {}: {} passed the {} in {}m, while we thought it is in {}m'
                         .format(cur_conv, bag, dv, dv_pos, o_pos), True)
                # model.shift(dv_pos - o_pos)

        if next_conv != cur_conv:
            self.log('{} KICKS {}'.format(dv, bag))
            bag_, evs = self.removeBagFromConv(cur_conv, bag.id)
            assert bag == bag_, "some other bag is here!!"

            evs += [DiverterKickAction()]
            evs += self.putBagToConv(next_conv, bag, 0)
            return evs
        else:
            self.log('{} PASSES {}'.format(dv, bag))
            return []

    def finishBag(self, sink: AgentId, bag: Bag) -> List[WorldEvent]:
        self.log('bag {} is OUT'.format(bag))
        assert bag.dst == sink, "NOT OUR DST!"

        evs = []
        self.current_bags.pop(bag.id)

        if bag.id in self.sink_bags[sink]:
            self.sink_bags[sink].remove(bag.id)
        else:
            prev_conv = self.bag_convs[bag.id]
            model = self.conveyor_models[prev_conv]
            o_pos = model.objPos(bag.id)
            if o_pos != model.length:
                self.log('DIVERGENCE WITH REALITY: conv {}: {} reached the {} in {}m, while we thought it is in {}m'
                         .format(prev_conv, bag, sink, model.length, o_pos), True)
                # model.shift(model.length - o_pos)

            _, evs = self.removeBagFromConv(prev_conv, bag.id)

        return evs + [BagReceiveAction(bag)]

    def routeBag(self, node: AgentId, bag: Bag, preview=False) -> AgentId:
        """
        All routing strategy is here. Override in subclasses
        """
        path = nx.dijkstra_path(self.topology, node, bag.dst, weight='length')
        return path[1]

    def putBagToConv(self, conv_idx, bag, pos) -> List[WorldEvent]:
        self.log('BAG {} -> CONV {} ({}m)'.format(bag, conv_idx, pos))
        self.conveyor_models[conv_idx].putObject(bag.id, bag, pos)
        self.bag_convs[bag.id] = conv_idx
        return self.start(conv_idx) + self._cancelDelayedStop(conv_idx)

    def removeBagFromConv(self, conv_idx, bag_id) -> Tuple[Bag, List[WorldEvent]]:
        self.log('BAG #{} LEAVES CONV {}'.format(bag_id, conv_idx))
        model = self.conveyor_models[conv_idx]
        bag = model.removeObject(bag_id)
        self.bag_convs.pop(bag.id)

        if len(model.objects) == 0:
            evs = self._scheduleDelayedStop(conv_idx)
        else:
            evs = []
        return bag, evs

    def leaveConvEnd(self, conv_idx, bag_id) -> List[WorldEvent]:
        bag, evs = self.removeBagFromConv(conv_idx, bag_id)
        up_node = conveyor_edges(self.topology, conv_idx)[-1][1]

        if agent_type(up_node) != 'sink':
            ps = self.topology.nodes[up_node]
            up_conv = ps['conveyor']
            up_pos = ps['conveyor_pos']
            return evs + self.putBagToConv(up_conv, bag, up_pos)
        else:
            self.sink_bags[up_node].add(bag.id)
            return evs

    def start(self, conv_idx, speed=None) -> List[WorldEvent]:
        if speed is None:
            speed = self.max_conv_speeds[conv_idx]
        return self.setSpeed(conv_idx, speed)

    def stop(self, conv_idx) -> List[WorldEvent]:
        return self.setSpeed(conv_idx, 0)

    def setSpeed(self, conv_idx, new_speed: float) -> List[WorldEvent]:
        model = self.conveyor_models[conv_idx]
        if model.speed != new_speed:
            model.setSpeed(new_speed)
            return [MasterEvent(('conveyor', conv_idx), ConveyorSpeedChangeAction(new_speed))]
        else:
            return []

    def _scheduleDelayedStop(self, conv_idx)  -> List[WorldEvent]:
        ev = self.delayed(self.stop_delay, lambda: self._withInterrupt(lambda: self.stop(conv_idx)))
        self.conv_delayed_stops[conv_idx] = ev.id
        return [ev]

    def _cancelDelayedStop(self, conv_idx) -> List[WorldEvent]:
        delay_id = self.conv_delayed_stops[conv_idx]
        if self.hasDelayed(delay_id):
            return [self.cancelDelayed(delay_id)]
        return []

    def _interrupt(self):
        for model in self.conveyor_models.values():
            model.pause()

        if self.hasDelayed(self.delayed_update):
            return [self.cancelDelayed(self.delayed_update)]
        return []

    def _update(self):
        for model in self.conveyor_models.values():
            if model.dirty():
                model.startResolving()

        evs = []
        for (conv_idx, (bag, node, delay)) in all_unresolved_events(self.conveyor_models):
            if node in self.current_bags[bag.id]:
                continue

            model = self.conveyor_models[conv_idx]
            self.log('conv {}: handling {} on {}'.format(conv_idx, bag, node))

            atype = agent_type(node)
            if atype == 'conv_end':
                evs += self.leaveConvEnd(conv_idx, bag.id)

            self.current_bags[bag.id].add(node)

        evs += self._cascadeSpeedUpdate()

        # for model in self.conveyor_models.values():
        #     if model.dirty():
        #         model.startResolving()

        for model in self.conveyor_models.values():
            if model.resolving():
                model.endResolving()
            model.resume()

        return evs + self._scheduleUpdate()

    def _pauseUpdate(self):
        for model in self.conveyor_models.values():
            model.pause()
        return self._update()

    def _scheduleUpdate(self):
        next_events = all_next_events(self.conveyor_models)
        if len(next_events) > 0:
            conv_idx, (bag, node, delay) = next_events[0]
            assert delay > 0, "chto za!!1"

            ev = self.delayed(delay, self._pauseUpdate)
            self.delayed_update = ev.id
            return [ev]
        else:
            return []

    def _withInterrupt(self, callback):
        evs = self._interrupt()
        evs += callback()
        return evs + self._update()

    def _cascadeSpeedUpdate(self) -> List[WorldEvent]:
        if self.last_cascade_update_time == self.env.time():
            return []

        for cid, model in self.conveyor_models.items():
            discount = 0.3 * (len(model.objects) / model.length)
            self.max_conv_speeds[cid] = self.max_speed - discount

        while True:
            conv_speed_changed = False

            for u, v in self._bfsFromSinks():
                cid = self.topology[u][v]['conveyor']
                max_speed = self._maxAllowedSectionSpeed(u, v, self.max_conv_speeds[cid])
                if max_speed < self.max_conv_speeds[cid]:
                    self.max_conv_speeds[cid] = max_speed
                    conv_speed_changed = True

                self.topology[u][v]['max_allowed_speed'] = self.max_conv_speeds[cid]

            if not conv_speed_changed:
                break

        evs = []
        for cid, speed in self.max_conv_speeds.items():
            model = self.conveyor_models[cid]
            if len(model.objects) > 0:
                if speed == 0 and model.speed > 0:
                    self.log('CLOGGED: conv {}\n  - objs: {}'
                             .format(cid, model.object_positions), True)
                elif speed > 0 and model.speed == 0:
                    self.log('UNCLOGGED: conv {}\n  - objs: {}'
                             .format(cid, model.object_positions), True)

                evs += self.setSpeed(cid, speed)

        self.last_cascade_update_time = self.env.time()
        return evs

    def _maxAllowedSectionSpeed(self, u, v, cur_max_speed=None):
        if cur_max_speed is None:
            cur_max_speed = self.max_speed

        exit_type = agent_type(v)

        if exit_type == 'sink':
            # sinks have no limitations
            return cur_max_speed

        conv_idx = self.topology[u][v]['conveyor']
        model = self.conveyor_models[conv_idx]
        u_conv = conveyor_idx(self.topology, u)
        v_conv = conveyor_idx(self.topology, v)

        u_pos = 0 if (u_conv != conv_idx or agent_type(u) == 'source') else model.checkpointPos(u)
        v_pos = model.length if v_conv != conv_idx else model.checkpointPos(v)

        nxt_exiting = model.nearestObject(v_pos, preference='prev')
        if nxt_exiting is not None:
            bag, bag_pos = nxt_exiting
            if bag_pos <= u_pos:
                # let that be handled by prev one
                return cur_max_speed

            dist_to_end = v_pos - bag_pos
            assert dist_to_end >= 0, \
                "searched for objs prev {}, found ({}, {})! ({} {})".format(v_pos, bag, bag_pos, u, v)

            if exit_type == 'diverter':
                try:
                    next_cp = self.routeBag(v, bag, preview=True)
                except:
                    self.log('edge ({}; {}m - {}; {}m): (#{}; {}m) cannot be routed'
                             .format(u, u_pos, v, v_pos, bag, bag_pos), True)
                    raise

                up_edge = self.topology[v][next_cp]
                up_conv = up_edge['conveyor']
                if up_conv == conv_idx:
                    # cannot do much about it
                    return cur_max_speed
                else:
                    up_model = self.conveyor_models[up_conv]
                    up_speed = up_edge['max_allowed_speed']
                    return find_max_speed(up_model, 0, up_speed, dist_to_end, cur_max_speed)
            else: # junction
                if v_conv == conv_idx:
                    return cur_max_speed
                else:
                    up_model = self.conveyor_models[v_conv]
                    up_pos = up_model.checkpointPos(v)
                    next_cp = next_same_conv_node(self.topology, v)
                    up_speed = self.topology[v][next_cp]['max_allowed_speed']
                    return find_max_speed(up_model, up_pos, up_speed, dist_to_end, cur_max_speed)
        else:
            return cur_max_speed

    def _bfsFromSinks(self):
        for u, v, _ in nx.edge_bfs(self.topology, list(self.sink_bags.keys()), orientation='reverse'):
            yield u, v


class CentralizedOracle(CentralizedController, Oracle):
    def __init__(self, conveyor_models: Dict[int, ConveyorModel], topology: nx.DiGraph,
                 max_speed: float, stop_delay: float, **kwargs):
        MasterHandler.__init__(self, **kwargs)
        self.topology = topology
        self.max_speed = max_speed
        self.stop_delay = stop_delay

        for u, v in self.topology.edges:
            self.topology[u][v]['max_allowed_speed'] = self.max_speed

        self.conveyor_models = conveyor_models

        conv_ids = list(conveyor_models.keys())
        self.max_conv_speeds = {cid: self.max_speed for cid in conv_ids}
        self.conv_delayed_stops = {cid: -1 for cid in conv_ids}
        self.last_cascade_update_time = -1
        self.sink_bags = {}
        for node in self.topology.nodes:
            if agent_type(node) == 'sink':
                self.sink_bags[node] = set()

    def handleSlaveEvent(self, slave_id: AgentId, event: WorldEvent) -> List[WorldEvent]:
        if isinstance(event, (IncomingBagEvent, OutgoingBagEvent, PassedBagEvent, BagDetectionEvent)):
            evs = []
            if isinstance(event, BagDetectionEvent):
                atype = agent_type(slave_id)
                if atype == 'diverter':
                    evs += self.divertBag(slave_id, event.bag)
                elif atype == 'sink':
                    evs += self.bagArrived(slave_id, event.bag)
            elif isinstance(event, IncomingBagEvent):
                evs += self.incomingBag(slave_id[1], event.sender, event.bag, event.node)
            elif isinstance(event, OutgoingBagEvent):
                evs += self.outgoingBag(slave_id[1], event.bag, event.node)
            elif isinstance(event, PassedBagEvent):
                evs += self.passedBag(slave_id[1], event.bag, event.node)

            return evs + self._cascadeSpeedUpdate()
        else:
            return super().handleSlaveEvent(slave_id, event)

    def divertBag(self, dv: AgentId, bag: Bag) -> List[WorldEvent]:
        nxt = self.routeBag(dv, bag)
        cur_conv = conveyor_idx(self.topology, dv)
        next_conv = self.topology[dv][nxt]['conveyor']

        if next_conv != cur_conv:
            self.log('{} KICKS {}'.format(dv, bag))
            return [DiverterKickAction()]
        else:
            self.log('{} PASSES {}'.format(dv, bag))
            return []

    def bagArrived(self, sink, bag):
        self.log('bag {} is OUT'.format(bag))
        assert bag.dst == sink, "NOT OUR DST!"
        return [BagReceiveAction(bag)]

    def incomingBag(self, conv_idx, sender, bag, node):
        return self.start(conv_idx) + self._cancelDelayedStop(conv_idx)

    def outgoingBag(self, conv_idx, bag, node):
        if len(self.conveyor_models[conv_idx].objects) == 0:
            return self._scheduleDelayedStop(conv_idx)
        return []

    def passedBag(self, conv_idx, bag, node):
        return []

    def setSpeed(self, conv_idx, new_speed: float) -> List[WorldEvent]:
        model = self.conveyor_models[conv_idx]
        if model.speed != new_speed:
            return [MasterEvent(('conveyor', conv_idx), ConveyorSpeedChangeAction(new_speed))]
        return []

    def _withInterrupt(self, callback):
        evs = callback()
        return evs + self._cascadeSpeedUpdate()

    def _update(self):
        raise NotImplementedError()

    def _interrupt(self):
        raise NotImplementedError()
