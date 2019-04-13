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
            res = self.processNewAnnouncement(msg.node, msg.state)

            # Do some action if initial announcements exchange is complete
            if self.networkComplete():
                self.networkInit()
            return res
        return False

    def networkComplete(self):
        """
        Never call `networkInit` by default
        """
        return False

    def networkInit(self):
        raise NotImplementedError()

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
        for (m, params) in adj_links.items():
            self.network.add_edge(self.id, m, **params)

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

class LSConveyorMixin(object):
    """
    Mixin for state routers which are working in a conveyor
    environment. Does not inherit `LinkStateRouter` in order
    to maintain sane parent's MRO. Only for usage as a mixin
    in other classes.
    """

    def __init__(self, env: DynamicEnv, **kwargs):
        super().__init__(env, **kwargs)
        self.network.nodes[self.id]['works'] = False

    def _conveyorWorks(self) -> bool:
        return self.network.nodes[self.id]['works']

    def _setConveyorWorkStatus(self, works: bool) -> List[Message]:
        if self._conveyorWorks() != works:
            self.network.nodes[self.id]['works'] = works
            return self._announceState()
        return []

    def route(self, sender: int, pkg: Package) -> Tuple[int, List[Message]]:
        """
        Makes sure that bags are not sent to the path which can not lead to
        the destination
        """
        old_neighbours = self.out_neighbours
        filter_func = lambda v: nx.has_path(self.network, v, pkg.dst)
        self.out_neighbours = set(filter(filter_func, old_neighbours))

        to, msgs = super().route(sender, pkg)
        scheduled_stop_time = self.env.time() + self.env.stop_delay()
        msgs.append(OutConveyorMsg(StopTimeUpdMsg(scheduled_stop_time)))

        self.out_neighbours = old_neighbours
        return to, msgs

    def handleServiceMsg(self, sender: int, msg: ServiceMessage) -> List[Message]:
        if isinstance(msg, ConveyorServiceMsg):
            if isinstance(msg, ConveyorStartMsg):
                self.scheduled_stop_time = self.env.time()
                return self._setConveyorWorkStatus(True)
            elif isinstance(msg, ConveyorStopMsg):
                return self._setConveyorWorkStatus(False)
            return []
        else:
            return super().handleServiceMsg(sender, msg)

    def getState(self):
        links = super().getState()
        return {'links': links, 'works': self._conveyorWorks()}

    def processNewAnnouncement(self, node: int, state) -> bool:
        links_ok = super().processNewAnnouncement(node, state['links'])
        self.network.nodes[node]['works'] = state['works']
        return links_ok

class LinkStateRouterConveyor(LSConveyorMixin, LinkStateRouter):
    pass
