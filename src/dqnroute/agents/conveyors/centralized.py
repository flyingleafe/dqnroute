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
        self.current_bags = {}

    def handleSlaveEvent(self, slave_id: AgentId, event: WorldEvent) -> List[WorldEvent]:
        if isinstance(event, BagDetectionEvent):
            bag = event.bag
            atype = agent_type(slave_id)

            evs = [] # self._interruptUpdate()

            if atype == 'source':
                evs += self.introduceBag(slave_id, bag)
            elif atype == 'diverter':
                evs += self.divertBag(slave_id, bag)
            elif atype == 'sink':
                evs += self.finishBag(bag)
            else:
                raise Exception('Bag detected on {} somehow!'.format(slave_id))

            return evs # + self._update()
        else:
            return super().handleSlaveEvent(slave_id, event)

    def introduceBag(self, src: AgentId, bag: Bag) -> List[WorldEvent]:
        self.current_bags[bag.id] = {}
        evs = []
        for cid in self.conveyor_models.keys():
            evs += self.start(cid) + self._cancelDelayedStop(cid)
        return evs

    def divertBag(self, dv: AgentId, bag: Bag) -> List[WorldEvent]:
        path = nx.dijkstra_path(self.topology, dv, bag.dst, weight='length')
        nxt = path[1]
        if conveyor_idx(self.topology, dv) != self.topology[dv][nxt]['conveyor']:
            return [DiverterKickAction()]
        return []

    def finishBag(self, bag: Bag) -> List[WorldEvent]:
        evs = []
        self.current_bags.pop(bag.id)
        if len(self.current_bags) == 0:
            for cid in self.conveyor_models.keys():
                evs += self._scheduleDelayedStop(cid)
        return evs + [BagReceiveAction(bag)]

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
            print('SETTING SPEED: conv {} -> {}m/s'.format(conv_idx, new_speed))
            return [MasterEvent(('conveyor', conv_idx), ConveyorSpeedChangeAction(new_speed))]
        else:
            return []

    def _scheduleDelayedStop(self, conv_idx)  -> List[WorldEvent]:
        ev = self.delayed(self.stop_delay, lambda: self.stop(conv_idx))
        self.conv_delayed_stops[conv_idx] = ev.id
        return [ev]

    def _cancelDelayedStop(self, conv_idx) -> List[WorldEvent]:
        delay_id = self.conv_delayed_stops[conv_idx]
        if self.hasDelayed(delay_id):
            return [self.cancelDelayed(delay_id)]
        return []
