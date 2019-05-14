from typing import List, Tuple
from .common import *
from ..base import *
from ...messages import *
from ...utils import *


class SimpleConveyor(Conveyor):
    """
    Simple conveyor which stops after some time has passed
    since the last bag exited it. It is connected to routers
    (diverters) seated on it and notifies them when it starts or stops.
    """
    def __init__(self, max_speed: float, stop_delay: float, **kwargs):
        super().__init__(**kwargs)
        self.stop_delay = stop_delay
        self.max_speed = max_speed

        self.current_bags = set()
        self.speed = 0
        self.delayed_stop = None
        self.ann_seq = 0

        self.env.register('stop_delay', lambda: self.stop_delay)
        self.env.register('scheduled_stop', lambda: self.scheduled_stop)

    def handleIncomingBag(self, notifier: AgentId, bag: Bag) -> List[WorldEvent]:
        self.current_bags.add(bag.id)
        if self.speed == 0:
            self.speed = self.max_speed
            return [ConveyorSpeedChangeAction(self.max_speed)] + self._announceSpeed()
        else:
            return self._cancelDelayedStop()

    def handleOutgoingBag(self, notifier: AgentId, bag: Bag) -> List[WorldEvent]:
        self.current_bags.remove(bag.id)
        if len(self.current_bags) == 0:
            ev = self.delayed(self.stop_delay, self.stop)
            self.delayed_stop = ev.id
            return [ev]
        return []

    def handlePassedBag(self, notifier: AgentId, bag: Bag) -> List[WorldEvent]:
        return []

    def stop(self) -> List[WorldEvent]:
        self.speed = 0
        self.delayed_stop = None
        return [ConveyorSpeedChangeAction(0)] + self._announceSpeed()

    def _cancelDelayedStop(self) -> List[WorldEvent]:
        if self.delayed_stop is not None:
            ev = self.cancelDelayed(self.delayed_stop)
            self.delayed_stop = None
            return [ev]
        return []

    def _announceSpeed(self) -> List[Message]:
        self.ann_seq += 1
        return self._notifyNbrs(StateAnnouncementMsg(self.id, self.ann_seq, self.speed))

    def _notifyNbrs(self, msg: Message) -> List[Message]:
        return [OutMessage(self.id, nbr, msg) for nbr in self.interface_map.values()]


class SimpleRouterConveyor(SimpleConveyor, RouterContainer):
    """
    Simple conveyor which also handles the logic of virtual junction routers.
    """
    def __init__(self, **kwargs):
        self.bag_statuses = {}

    def handleIncomingBag(self, notifier: AgentId, bag: Bag) -> List[WorldEvent]:
        evs = super().handleIncomingBag(notifier, bag)
        assert bag.id not in self.bag_statuses, \
            "Existing bag comes onto conveyor again somehow!"

        n_type = agent_type(notifier)
        bag_status = {'pos': -1, 'last_updated': self.env.time()}

        if n_type in ('source', 'diverter'):
            # sources and diverters are always in the beginning
            bag_status['pos'] = 0
            prev_node = notifier
        elif n_type == 'conveyor':
            pass
        else:
            raise Exception('Only source, diverter or conveyor can be an outside source of bag!')

        self.bag_statuses[bag.id] = bag_status
