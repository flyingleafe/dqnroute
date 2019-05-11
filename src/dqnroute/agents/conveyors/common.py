from typing import List, Tuple
from ..base import *
from ...messages import *
from ...utils import *

class ItemSource(MessageHandler):
    """
    Class which implements a bag source controller, which notifies
    the system about a new bag arrival.
    """
    def handleEvent(self, event: WorldEvent) -> List[WorldEvent]:
        if isinstance(event, BagDetectionEvent):
            nbr = self.interface_map[0]    # source is always connected only to upstream conv
            return [OutMessage(self.id, nbr, IncomingBagMsg(event.bag))]
        else:
            raise UnsupportedEventType(event)

    def handleMsgFrom(self, sender: AgentId, msg: Message) -> List[WorldEvent]:
        """
        Source controller ignores all incoming messages
        """
        return []


class ItemSink(MessageHandler):
    """
    Class which implements a sink controller, which detects
    an exit of a bag from the system
    """
    def handleEvent(self, event: WorldEvent) -> List[WorldEvent]:
        if isinstance(event, BagDetectionEvent):
            nbr = self.interface_map[0]     # sink is only connected to downstream conv
            return [BagReceiveAction(event.bag), OutMessage(self.id, nbr, OutgoingBagMsg(event.bag))]
        else:
            raise UnsupportedEventType(event)

    def handleMsgFrom(self, sender: AgentId, msg: Message) -> List[WorldEvent]:
        """
        Sink controller ignores all incoming messages
        """
        return []
