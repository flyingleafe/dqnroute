import random
import pprint
import networkx as nx

from copy import deepcopy
from typing import List, Tuple, Dict
from ..base import *
from ...messages import *
from ...constants import INFTY

class AbstractStateHandler(MessageHandler):
    """
    A router which implements a link-state protocol but the state is
    not necessarily link-state and can be abstracted out.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.seq_num = 0
        self.announcements = {}

    def init(self, config) -> List[WorldEvent]:
        msgs = super().init(config)
        return msgs + self._announceState()

    def handleMsgFrom(self, sender: AgentId, msg: Message) -> List[WorldEvent]:
        if isinstance(msg, StateAnnouncementMsg):
            if msg.node == self.id:
                return []

            if msg.node not in self.announcements or self.announcements[msg.node].seq < msg.seq:
                self.announcements[msg.node] = msg
                broadcast, msgs = self.processNewAnnouncement(msg.node, msg.state)
                self.networkStateChanged()

                if broadcast:
                    msgs += self.broadcast(msg, exclude=[sender])
                return msgs
            return []

        else:
            return super().handleMsgFrom(sender, msg)

    def _announceState(self) -> List[Message]:
        state = self.getState()
        if state is None:
            return []

        announcement = StateAnnouncementMsg(self.id, self.seq_num, state)
        self.seq_num += 1
        return self.broadcast(announcement)

    def networkStateChanged(self):
        """
        Check if relevant network state has been changed and perform
        some action accordingly.
        Do nothing by default; should be overridden in subclasses.
        """
        pass

    def getState(self):
        """
        Should be overridden by subclasses. If returned state is `None`,
        no announcement is made.
        """
        raise NotImplementedError()

    def processNewAnnouncement(self, node: int, state) -> Tuple[bool, List[WorldEvent]]:
        raise NotImplementedError()

class LinkStateRouter(Router, AbstractStateHandler):
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

    def removeLink(self, to: AgentId) -> List[Message]:
        msgs = super().removeLink(to)
        self.network.remove_edge(to, self.id)
        self.network.remove_edge(self.id, to)
        return msgs + self._announceState()

    def route(self, sender: AgentId, pkg: Package, allowed_nbrs: List[AgentId]) -> Tuple[AgentId, List[Message]]:
        if len(allowed_nbrs) == 1:
            return allowed_nbrs[0], []

        path = nx.dijkstra_path(self.network, self.id, pkg.dst, weight=self.edge_weight)
        if path[1] in allowed_nbrs:
            return path[1], []

        else:
            min_nbr = None
            min_len = INFTY
            for nbr in allowed_nbrs:
                elen = self.network[self.id][nbr][self.edge_weight]
                plen = nx.dijkstra_path_length(self.network, nbr, pkg.dst, weight=self.edge_weight)
                if elen + plen < min_len:
                    min_nbr = nbr
                    min_len = elen + plen
            assert min_nbr is not None, "!!sdfsdfs!!!"
            return min_nbr, []

    def pathCost(self, to: AgentId, through=None) -> float:
        if through is None:
            return nx.dijkstra_path_length(self.network, self.id, to, weight=self.edge_weight)
        else:
            l1 = nx.dijkstra_path_length(self.network, self.id, through, weight=self.edge_weight)
            l2 = nx.dijkstra_path_length(self.network, through, self.id, weight=self.edge_weight)
            return l1 + l2

    def getState(self):
        return self.network.adj[self.id]

    def processNewAnnouncement(self, node: AgentId, neighbours) -> Tuple[bool, List[WorldEvent]]:
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

        return changed, []


class LSConveyorMixin(object):
    """
    Mixin for state routers which are working in a conveyor
    environment. Does not inherit `LinkStateRouter` in order
    to maintain sane parent's MRO. Only for usage as a mixin
    in other classes.
    """

    def __init__(self, conv_stop_delay: float, **kwargs):
        super().__init__(**kwargs)
        self.conv_stop_delay = conv_stop_delay
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

    def detectEnqueuedPkg(self, sender: AgentId, pkg: Package) -> List[WorldEvent]:
        msgs = super().detectEnqueuedPkg(sender, pkg)

        allowed_nbrs = only_reachable(self.network, pkg.dst, self.network.successors(self.id))
        to, _ = self.route(sender, pkg, allowed_nbrs)
        if isinstance(self, RewardAgent):
            self._pending_pkgs.pop(pkg.id)

        return msgs + [PkgRoutePredictionAction(to, pkg)]

    def handleMsgFrom(self, sender: AgentId, msg: Message) -> List[Message]:
        if isinstance(msg, ConveyorStartMsg):
            return self._setConveyorWorkStatus(True)
        elif isinstance(msg, ConveyorStopMsg):
            return self._setConveyorWorkStatus(False)
        else:
            return super().handleMsgFrom(sender, msg)

    def getState(self):
        sub = super().getState()
        return {'sub': sub, 'works': self._conveyorWorks()}

    def processNewAnnouncement(self, node: int, state) -> Tuple[bool, List[WorldEvent]]:
        sub_ok, msgs = super().processNewAnnouncement(node, state['sub'])
        works_changed = self._conveyorWorks(node) != state['works']
        self.network.nodes[node]['works'] = state['works']
        return sub_ok or works_changed, msgs

class LinkStateRouterConveyor(LSConveyorMixin, LinkStateRouter):
    pass
