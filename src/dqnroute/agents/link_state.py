import random
import networkx as nx

from copy import deepcopy
from typing import List, Tuple, Dict
from .base import *
from ..messages import *

class LinkStateRouter(Router):
    """
    A router which implements simple link-state algorithm
    """
    def __init__(self, env: DynamicEnv, network: nx.DiGraph, **kwargs):
        super().__init__(env, **kwargs)
        self.network = network
        self.seq_num = 0
        self.announcements = {}

    def init(self, config) -> List[Message]:
        msgs = super().init(config)
        return msgs + self._announceLinkState()

    def addLink(self, to: int, params={}) -> List[Message]:
        msgs = super().addLink(to, params)
        self.network.add_edge(self.id, to, **params)
        return msgs + self._announceLinkState()

    def removeLink(self, to: int) -> List[Message]:
        msgs = super().removeLink(to)
        self.network.remove_edge(self.id, to)
        return msgs + self._announceLinkState()

    def route(self, sender: int, pkg: Package) -> Tuple[int, List[Message]]:
        paths = nx.all_shortest_paths(self.network, self.id, pkg.dst,
                                      weight='latency')
        return random.choice(list(paths))[1], []

    def handleServiceMsg(self, sender: int, msg: ServiceMessage) -> List[Message]:
        if isinstance(msg, LSAnnouncementMsg):
            if self._processLSAnnouncement(msg.node, msg.seq, msg.neighbours):
                return [OutMessage(self.id, v, deepcopy(msg))
                        for v in (self.all_neighbours - set([sender]))]
            return []

        else:
            return super().handleServiceMsg(sender, msg)

    def _announceLinkState(self) -> List[Message]:
        neighbour_links = self.network.adj[self.id]
        announcement = LSAnnouncementMsg(self.id, self.seq_num, neighbour_links)
        self.seq_num += 1
        return [OutMessage(self.id, v, deepcopy(announcement)) for v in self.all_neighbours]

    def _processLSAnnouncement(self, node: int, seq: int, neighbours) -> bool:
        if node not in self.announcements or self.announcements[node]["seq"] < seq:
            self.announcements[node] = {"seq": seq, "neighbours": neighbours}

            for (m, params) in neighbours.items():
                self.network.add_edge(node, m, **params)

            for m in set(self.network.nodes()) - set(neighbours.keys()):
                try:
                    self.network.remove_edge(node, m)
                except nx.NetworkXError:
                    pass

            return True
        return False
