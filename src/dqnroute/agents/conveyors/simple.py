from typing import List, Tuple
from ..base import *
from ...messages import *
from ...utils import *

class SimpleConveyor(Conveyor):
    """
    Simple conveyor which stops after some time has passed
    since the last bag exited it. It is connected to routers
    (diverters) seated on it and notifies them when it starts or stops.
    """
    def __init__(self, env: DynamicEnv, routers: List[int],
                 stop_delay: float, **kwargs):
        super().__init__(env, **kwargs)
        self.stop_delay = stop_delay
        self.routers = routers

        self.current_bags = set()
        self.is_working = False
        self.delayed_stop = False
        self.scheduled_stop = 0

        self.env.register('stop_delay', lambda: self.stop_delay)
        self.env.register('scheduled_stop', lambda: self.scheduled_stop)

    def start(self) -> List[Message]:
        if not self.is_working:
            self.is_working = True
            self.scheduled_stop = self.env.time()
            return self._notifyRouters(ConveyorStartMsg())
        else:
            return self._cancelDelayedStop()
        return []

    def stop(self) -> List[Message]:
        if self.is_working:
            self.is_working = False
            self.delayed_stop = False
            return self._notifyRouters(ConveyorStopMsg())
        return []

    def handleIncomingBag(self, bag: Bag) -> List[Message]:
        self.current_bags.add(bag.id)
        if not self.is_working:
            return [OutConveyorMsg(ConveyorStartMsg())]
        else:
            return self._cancelDelayedStop()

    def handleOutgoingBag(self, bag: Bag) -> List[Message]:
        self.current_bags.remove(bag.id)
        if len(self.current_bags) == 0:
            self.delayed_stop = True
            return [DelayedMessage(self.id, self.stop_delay,
                                   OutConveyorMsg(ConveyorStopMsg()))]
        return []

    def handleCustomMsg(self, msg: ConveyorMessage) -> List[Message]:
        if isinstance(msg, StopTimeUpdMsg):
            self.scheduled_stop = msg.time
            return []
        else:
            return super().handleCustomMsg(msg)

    def _cancelDelayedStop(self) -> List[Message]:
        if self.delayed_stop:
            self.delayed_stop = False
            return [DelayInterruptMessage(self.id)]
        return []

    def _notifyRouters(self, msg: Message) -> List[Message]:
        return [OutMessage(-1, rid, msg) for rid in self.routers]
