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
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.seq_num = 0
        self.announcements = {}

    def init(self, config) -> List[Message]:
        msgs = super().init(config)
        return msgs + self._announceState()

    def handleMsgFrom(self, sender: AgentId, msg: Message) -> List[Message]:
        if isinstance(msg, StateAnnouncementMsg):
            if self._processStateAnnouncement(msg):
                return [OutMessage(self.id, v, deepcopy(msg))
                        for v in (self.all_neighbours - set([sender]))]
            return []

        else:
            return super().handleMsgFrom(sender, msg)

    def _announceState(self) -> List[Message]:
        state = self.getState()
        announcement = StateAnnouncementMsg(self.id, self.seq_num, state)
        self.seq_num += 1
        return [OutMessage(self.id, v, deepcopy(announcement)) for v in self.all_neighbours]

    def _processStateAnnouncement(self, msg: StateAnnouncementMsg) -> bool:
        if msg.node == self.id:
            return False

        if msg.node not in self.announcements or self.announcements[msg.node].seq < msg.seq:
            self.announcements[msg.node] = msg
            res = self.processNewAnnouncement(msg.node, msg.state)
            self.networkStateChanged()
            return res
        return False

    def networkStateChanged(self):
        """
        Check if relevant network state has been changed and perform
        some action accordingly.
        Do nothing by default; should be overridden in subclasses.
        """
        pass

    def getState(self):
        raise NotImplementedError()

    def processNewAnnouncement(self, node: int, state) -> bool:
        raise NotImplementedError()

class LinkStateRouter(AbstractLinkStateRouter):
    """
    Simple link-state router
    """
    def __init__(self, adj_links, edge_weight='weight', **kwargs):
        super().__init__(**kwargs)
        self.network = nx.DiGraph()
        self.edge_weight = edge_weight

        self.network.add_node(self.id)
        for (m, params) in adj_links.items():
            self.network.add_edge(self.id, m, **params)

    def addLink(self, to: AgentId, params={}) -> List[Message]:
        msgs = super().addLink(to, params)
        self.network.add_edge(to, self.id, **params)
        self.network.add_edge(self.id, to, **params)
        return msgs + self._announceState()

    def removeLink(self, to: int) -> List[Message]:
        msgs = super().removeLink(to)
        self.network.remove_edge(to, self.id)
        self.network.remove_edge(self.id, to)
        return msgs + self._announceState()

    def route(self, sender: AgentId, pkg: Package) -> Tuple[AgentId, List[Message]]:
        path = nx.dijkstra_path(self.network, self.id, pkg.dst,
                                weight=self.edge_weight)
        return path[1], []

    def getState(self):
        return self.network.adj[self.id]

    def processNewAnnouncement(self, node: AgentId, neighbours) -> bool:
        changed = False

        for (m, params) in neighbours.items():
            if self.network.get_edge_data(node, m) != params:
                self.network.add_edge(node, m, **params)
                changed = True

        for m in set(self.network.nodes()) - set(neighbours.keys()):
            try:
                self.network.remove_edge(node, m)
                changed = True
            except nx.NetworkXError:
                pass

        return changed


class LSConveyorMixin(object):
    """
    Mixin for state routers which are working in a conveyor
    environment. Does not inherit `LinkStateRouter` in order
    to maintain sane parent's MRO. Only for usage as a mixin
    in other classes.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.network.nodes[self.id]['works'] = False

    def _conveyorWorks(self, node=None) -> bool:
        if node is None:
            node = self.id
        return self.network.nodes[node].get('works', False)

    def _setConveyorWorkStatus(self, works: bool) -> List[Message]:
        if self._conveyorWorks() != works:
            self.network.nodes[self.id]['works'] = works
            return self._announceState()
        return []

    def route(self, sender: AgentId, pkg: Package) -> Tuple[AgentId, List[Message]]:
        """
        Makes sure that bags are not sent to the path which can not lead to
        the destination
        """
        old_neighbours = self.out_neighbours
        self.out_neighbours = set(only_reachable(self.network, pkg.dst, old_neighbours))

        to, msgs = super().route(sender, pkg)
        scheduled_stop_time = self.env.time() + self.env.stop_delay()
        msgs.append(OutConveyorMsg(StopTimeUpdMsg(scheduled_stop_time)))

        self.out_neighbours = old_neighbours
        return to, msgs

    def handleMsgFrom(self, sender: AgentId, msg: Message) -> List[Message]:
        if isinstance(msg, ConveyorServiceMsg):
            if isinstance(msg, ConveyorStartMsg):
                self.scheduled_stop_time = self.env.time()
                return self._setConveyorWorkStatus(True)
            elif isinstance(msg, ConveyorStopMsg):
                return self._setConveyorWorkStatus(False)
            return []
        else:
            return super().handleMsgFrom(sender, msg)

    def getState(self):
        sub = super().getState()
        return {'sub': sub, 'works': self._conveyorWorks()}

    def processNewAnnouncement(self, node: int, state) -> bool:
        sub_ok = super().processNewAnnouncement(node, state['sub'])
        works_changed = self._conveyorWorks(node) != state['works']
        self.network.nodes[node]['works'] = state['works']
        return sub_ok or works_changed

class LinkStateRouterConveyor(LSConveyorMixin, LinkStateRouter):
    pass
