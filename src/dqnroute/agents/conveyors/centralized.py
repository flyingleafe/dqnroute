import networkx as nx

from typing import List, Tuple, Dict
from ..base import *
from ...messages import *
from ...utils import *
from ...conveyor_model import *

BAG_RADIUS = 1
SPEED_STEP = 0.1
SPEED_ROUND_DIGITS = 5

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

        self.conveyor_models = {}
        for (conv_id, length) in conv_lengths.items():
            checkpoints = conveyor_adj_nodes(self.topology, conv_id,
                                             only_own=True, data='conveyor_pos')
            model = ConveyorModel(length, self.max_speed, checkpoints,
                                  model_id=('conveyor', conv_id))
            self.conveyor_models[conv_id] = model

        conv_ids = list(conv_lengths.keys())
        self.conv_delayed_stops = {cid: -1 for cid in conv_ids}
        self.delayed_update = -1
        self.current_bags = {}
        self.bag_convs = {}

        self.conv_ups = {}
        self.prev_convs = {cid: set() for cid in conv_ids}
        self.sink_convs = []
        self.sink_bags = {}
        for cid in conv_ids:
            es = conveyor_edges(self.topology, cid)
            up_node = es[-1][1]
            self.conv_ups[cid] = up_node
            if agent_type(up_node) == 'sink':
                self.sink_convs.append(cid)
                self.sink_bags[up_node] = set()
            else:
                up_conv = conveyor_idx(self.topology, up_node)
                self.prev_convs[up_conv].add(cid)

        evs = self._update()
        assert len(evs) == 0, "doing something on first update? hey!"

    def log(self, msg, force=False):
        super().log(msg, force)

    def handleSlaveEvent(self, slave_id: AgentId, event: WorldEvent) -> List[WorldEvent]:
        if isinstance(event, BagDetectionEvent):
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

        if next_conv != cur_conv:
            bag_, evs = self.removeBagFromConv(cur_conv, bag.id)
            assert bag == bag_, "some other bag is here!!"

            evs += [DiverterKickAction()]
            evs += self.putBagToConv(next_conv, bag, 0)
            return evs
        else:
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
            _, evs = self.removeBagFromConv(prev_conv, bag.id)

        return evs + [BagReceiveAction(bag)]

    def routeBag(self, node: AgentId, bag: Bag) -> AgentId:
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
        up_node = self.conv_ups[conv_idx]

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
            speed = self._maxAllowedSpeed(conv_idx)
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
            model.pause(self.env.time())

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
            # elif atype == 'diverter':
                # self.passToAgent(node, BagDetectionEvent(bag))

            self.current_bags[bag.id].add(node)

        evs += self._cascadeSpeedUpdate()

        for model in self.conveyor_models.values():
            if model.dirty():
                model.startResolving()

        for model in self.conveyor_models.values():
            if model.resolving():
                model.endResolving()
            model.resume(self.env.time())

        return evs + self._scheduleUpdate()

    def _pauseUpdate(self):
        for model in self.conveyor_models.values():
            model.pause(self.env.time())
        return self._update()

    def _scheduleUpdate(self):
        next_events = all_next_events(self.conveyor_models)
        if len(next_events) > 0:
            conv_idx, (bag, node, delay) = next_events[0]

            assert delay >= 0, "chto za!!1"

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
        queue = list(self.sink_convs)
        seen = set()
        evs = []
        while len(queue) > 0:
            cid = queue.pop(0)
            seen.add(cid)
            model = self.conveyor_models[cid]

            max_speed = self._maxAllowedSpeed(cid)
            if len(model.objects) > 0:
                assert not self.hasDelayed(self.conv_delayed_stops[cid]), "HHEEEET"
                evs += self.setSpeed(cid, max_speed)

            for pr in self.prev_convs[cid] - seen:
                queue.append(pr)
        return evs

    def _maxAllowedSpeed(self, conv_idx):
        speed = self.max_speed
        model = self.conveyor_models[conv_idx]
        up_node = self.conv_ups[conv_idx]

        if agent_type(up_node) != 'sink':
            up_conv = conveyor_idx(self.topology, up_node)
            up_model = self.conveyor_models[up_conv]
            up_pos = up_model.checkpointPos(up_node)

            if len(up_model.objects) > 0:
                while speed > 0:
                    leave_time = model.timeTillObjectLeave(speed)
                    if leave_time is None:
                        break

                    last_bag, last_pos = model.object_positions[-1]

                    bag, pos = up_model.nearestObject(up_pos, after=leave_time)
                    if abs(pos - up_pos) >= 2*BAG_RADIUS:
                        self.log('bag {}: conv {} -> ({}; {}) -> conv {}, nearest - (bag {}; {})'
                                 .format(last_bag, conv_idx, up_node, up_pos, up_conv, bag.id, pos))
                        break

                    self.log('conv {}, last bag ({}; {}m) goes to ({}; {}m):\n  - cannot set speed {} due to ({}; {}m) on conv {}'
                             .format(conv_idx, last_bag, last_pos, up_node, up_pos,
                                     speed, bag, pos, up_conv))
                    speed = round(speed - SPEED_STEP, SPEED_ROUND_DIGITS)

        return max(0, speed)
