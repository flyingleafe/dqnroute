import networkx as nx

from typing import List, Tuple, Dict
from ..base import *
from ...messages import *
from ...utils import *
from ...conveyor_model import *

BAG_RADIUS = 1

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
                evs += self.finishBag(bag)
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
        path = nx.dijkstra_path(self.topology, dv, bag.dst, weight='length')
        nxt = path[1]
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

    def finishBag(self, bag: Bag) -> List[WorldEvent]:
        self.log('bag {} is OUT'.format(bag))

        self.current_bags.pop(bag.id)
        prev_conv = self.bag_convs[bag.id]
        _, evs = self.removeBagFromConv(prev_conv, bag.id)
        return evs + [BagReceiveAction(bag)]

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
        up_node = conveyor_edges(self.topology, conv_idx)[-1][1]

        if agent_type(up_node) != 'sink':
            bag, evs = self.removeBagFromConv(conv_idx, bag_id)
            ps = self.topology.nodes[up_node]
            up_conv = ps['conveyor']
            up_pos = ps['conveyor_pos']
            return evs + self.putBagToConv(up_conv, bag, up_pos)
        return []

    def start(self, conv_idx, speed=None) -> List[WorldEvent]:
        if speed is None:
            speed = self.max_speed
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
            _, (_, _, delay) = next_events[0]
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
