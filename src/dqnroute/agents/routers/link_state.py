import random
import networkx as nx

from copy import deepcopy
from typing import List, Tuple, Dict
from ..base import *
from ...messages import *

class AbstractLinkStateRouter(Router):
    """
    A router which implements a link-state protocol where the notion
    of a link-state is abstracted out
    """
    def __init__(self, env: DynamicEnv, **kwargs):
        super().__init__(env, **kwargs)
        self.seq_num = 0
        self.announcements = {}

    def init(self, config) -> List[Message]:
        msgs = super().init(config)
        return msgs + self._announceState()

    def handleServiceMsg(self, sender: int, msg: ServiceMessage) -> List[Message]:
        if isinstance(msg, StateAnnouncementMsg):
            if self._processStateAnnouncement(msg):
                return [OutMessage(self.id, v, deepcopy(msg))
                        for v in (self.all_neighbours - set([sender]))]
            return []

        else:
            return super().handleServiceMsg(sender, msg)

    def _announceState(self) -> List[Message]:
        state = self.getState()
        announcement = StateAnnouncementMsg(self.id, self.seq_num, state)
        self.seq_num += 1
        return [OutMessage(self.id, v, deepcopy(announcement)) for v in self.all_neighbours]

    def _processStateAnnouncement(self, msg: StateAnnouncementMsg) -> bool:
        if msg.node not in self.announcements or self.announcements[msg.node].seq < msg.seq:
            self.announcements[msg.node] = msg
            return self.processNewAnnouncement(msg.node, msg.state)
        return False

    def getState(self):
        raise NotImplementedError()

    def processNewAnnouncement(self, node: int, state) -> bool:
        raise NotImplementedError()

class LinkStateRouter(AbstractLinkStateRouter):
    """
    Simple link-state router
    """
    def __init__(self, env: DynamicEnv, adj_links,
                 edge_weight='weight', **kwargs):
        super().__init__(env, **kwargs)
        self.network = nx.DiGraph()
        self.edge_weight = edge_weight

        self.network.add_node(self.id)
        self.processNewAnnouncement(self.id, adj_links)

    def addLink(self, to: int, direction: str, params={}) -> List[Message]:
        msgs = super().addLink(to, direction, params)
        if direction != 'out':
            self.network.add_edge(to, self.id, **params)
        if direction != 'in':
            self.network.add_edge(self.id, to, **params)
        return msgs + self._announceState()

    def removeLink(self, to: int, direction: str) -> List[Message]:
        msgs = super().removeLink(to, direction)
        if direction != 'out':
            self.network.remove_edge(to, self.id)
        if direction != 'in':
            self.network.remove_edge(self.id, to)
        return msgs + self._announceState()

    def route(self, sender: int, pkg: Package) -> Tuple[int, List[Message]]:
        path = nx.dijkstra_path(self.network, self.id, pkg.dst,
                                weight=self.edge_weight)
        return path[1], []

    def getState(self):
        return self.network.adj[self.id]

    def processNewAnnouncement(self, node: int, neighbours) -> bool:
        for (m, params) in neighbours.items():
            self.network.add_edge(node, m, **params)

        for m in set(self.network.nodes()) - set(neighbours.keys()):
            try:
                self.network.remove_edge(node, m)
            except nx.NetworkXError:
                pass

        return True

class LinkStateRouterConveyor(LinkStateRouter):
    def handleServiceMsg(self, sender: int, msg: ServiceMessage) -> List[Message]:
        if isinstance(msg, ConveyorServiceMsg):
            return []
        else:
            return super().handleServiceMsg(sender, msg)
