"""
Diverter class definitions
"""
import networkx as nx

from typing import List, Tuple
from .common import *
from ..base import *
from ...messages import *
from ...utils import *


class RouterDiverter(RouterContainer, Diverter, ConveyorStateHandler):
    """
    Diverter which uses logic of a given router class.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.predecessor = next(iter(self.topology.predecessors(self.id)))  # diverter has only one predecessor

    def handleMsgFrom(self, sender: AgentId, msg: Message) -> List[WorldEvent]:
        if isinstance(msg, DiverterNotification):
            self.log('notification request from {}'.format(sender))
            bag = msg.bag
            pkg = self.bagToPkg(bag)

            from_router = self.node_mapping[self.predecessor]
            router_id = self.routerId()
            msgs = self.handleViaRouter(router_id, PkgEnqueuedEvent(from_router, router_id, pkg))
            return [OutMessage(self.id, sender, msg)
                    if isinstance(msg, DiverterPrediction) else msg
                    for msg in msgs]
        else:
            return super().handleMsgFrom(sender, msg)

    def bagDetection(self, bag: Bag) -> List[WorldEvent]:
        """
        We redefine `bagDetection` instead of using `divert` to
        reuse action transformation logic from `RouterContainer`.
        """
        sender = self.node_mapping[self.predecessor]
        router_id = self.routerId()
        return self.handleBagViaRouter(sender, router_id, bag)

    def fromRouterAction(self, router_id: AgentId, action: Action) -> List[WorldEvent]:
        if isinstance(action, (PkgRouteAction, PkgRoutePredictionAction)):
            bag = self.pkgToBag(action.pkg)
            to_node = self.node_mapping_inv[action.to]
            self_conv = self.topology.nodes[self.id]['conveyor']
            to_conv = self.topology[self.id][to_node]['conveyor']
            kicked = self_conv != to_conv

            if isinstance(action, PkgRoutePredictionAction):
                return [DiverterPrediction(bag, kicked)]
            else:
                if kicked:
                    self.log('kicked bag #{} to conv {}: {} {}'.format(bag.id, to_conv, action.to, to_node))
                    evs = [DiverterKickAction()]
                    if not self._oracle:
                        evs += [OutMessage(self.id, ('conveyor', to_conv), IncomingBagMsg(bag)),
                                OutMessage(self.id, ('conveyor', self_conv), OutgoingBagMsg(bag))]
                    return evs
                else:
                    self.log('passed bag #{} to conv {}: {} {}'.format(bag.id, to_conv, action.to, to_node))
                    return [OutMessage(self.id, ('conveyor', self_conv), PassedBagMsg(bag))] if not self._oracle else []
        else:
            return super().fromRouterAction(router_id, action)

    def getState(self):
        return None
