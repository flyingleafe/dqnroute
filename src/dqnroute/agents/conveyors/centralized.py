import networkx as nx

from typing import List, Tuple
from ..base import *
from ...messages import *
from ...utils import *


class CentralizedController(MasterHandler):
    """
    A centralized controller for the whole conveyor system
    """
    def __init__(self, topology: nx.DiGraph, layout, **kwargs):
        super().__init__(**kwargs)
        self.topology = topology
        self.layout = layout

    def handleSlaveEvent(self, slave_id: AgentId, event: WorldEvent) -> List[WorldEvent]:
        if isinstance(event, BagDetectionEvent):
            bag = event.bag
            atype = agent_type(slave_id)

            if atype == 'source':
                self._introduceBag()
        else:
            return super().handleSlaveEvent(slave_id, event)
