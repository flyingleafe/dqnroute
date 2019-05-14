"""
Diverter class definitions
"""
import networkx as nx

from typing import List, Tuple
from .common import *
from ..base import *
from ...messages import *
from ...utils import *


class RouterDiverter(RouterContainer, Diverter):
    """
    Diverter which uses logic of a given router class.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.predecessor = next(iter(self.topology.predecessors(self.id)))  # diverter has only one predecessor

    def bagDetection(self, bag: Bag) -> List[WorldEvent]:
        """
        We redefine `bagDetection` instead of using `divert` to
        reuse action transformation logic from `RouterContainer`.
        """
        sender = self.node_mapping[self.predecessor]
        router_id = self.routerId()
        return self.handleViaRouter(router_id, PkgProcessingEvent(sender, router_id, bag))

    def handleMsgFrom(self, sender: AgentId, msg: Message) -> List[WorldEvent]:
        if isinstance(msg, IncomingBagMsg):
            sender = self.node_mapping[self.predecessor]
            router_id = self.routerId()
            return self.handleViaRouter(router_id, PkgEnqueuedEvent(sender, router_id, msg.bag))
        else:
            return super().handleMsgFrom(sender, msg)

    def fromRouterAction(self, router_id: AgentId, action: Action) -> List[WorldEvent]:
        if isinstance(action, PkgRouteAction):
            bag = action.pkg
            to_node = self.node_mapping_inv[event.to]
            self_conv = self.topology.nodes[self.id]['conveyor']
            to_conv = self.topology.nodes[to_node]['conveyor']

            if self_conv != to_conv:
                return [DiverterKickAction(),
                        OutMessage(self.id, ('conveyor', to_conv), IncomingBagMsg(bag)),
                        OutMessage(self.id, ('conveyor', self_conv), OutgoingBagMsg(bag))]
            else:
                return [OutMessage(self.id, ('conveyor', self_conv), PassedBagMsg(bag))]
        else:
            return super().fromRouterAction(router_id, action)
