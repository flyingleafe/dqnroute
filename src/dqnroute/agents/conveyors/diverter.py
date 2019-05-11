"""
Diverter class definitions
"""
import networkx as nx

from typing import List, Tuple
from ..base import *
from ..routers import LinkStateRouter
from ...messages import *
from ...utils import *


class RouterDiverter(Diverter):
    """
    Diverter which uses logic of a given router class
    """
    def __init__(self, env: DynamicEnv, neighbours: List[AgentId],
                 topology_graph: nx.DiGraph,
                 RouterClass, router_args, **kwargs):
        assert issubclass(RouterClass, Router), \
            "Given class is not a subclass of Router!"
        super().__init__(env=env, neighbours=neighbours, **kwargs)

        self.topology_graph = topology_graph
        self.node_mapping = {}
        self.node_mapping_inv = {}
        for (i, aid) in enumerate(sorted(topology_graph.nodes)):
            rid = ('router', i)
            self.node_mapping[aid] = rid
            self.node_mapping_inv[rid] = aid

        G = nx.relabel_nodes(topology_graph, self.node_mapping)
        self_rid = self.node_mapping[self.id]
        fake_nbrs = [v for (_, v) in G.to_undirected().edges(self_rid)]

        kwargs = make_router_cfg(G, self_rid)
        kwargs.update(router_args)
        if issubclass(RouterClass, LinkStateRouter):
            kwargs['adj_links'] = G.adj[self_rid]

        self.router = RouterClass(env=env, id=self_rid, neighbours=fake_nbrs,
                                  edge_weight='length', **kwargs)

    def divert(self, bag: Bag) -> Tuple[bool, List[Message]]:
        # TODO: do stuff
        return False, []

    def handleMsgFrom(self, sender: AgentId, msg: Message) -> List[WorldEvent]:
        if isinstance(msg, WrappedRouterMsg):
            router_msg = msg.inner
            msgs = self.router.handleMsgFrom(self.node_mapping[sender],
                                             self.toRouterMsg(msg))
            return [self.fromRouterMsg(m) for m in msgs]
        else:
            return super().handleMsgFrom(sender, msg)

    def wrapRouterMsg(self, msg: Message) -> Message:
        if isinstance(msg, DelayedMessage):
            return DelayedMessage(msg.id, msg.delay, self.wrapRouterMsg(msg.inner_msg))
        elif isinstance(msg, DelayInterruptMessage):
            return msg
        else:
            return WrappedRouterMsg(msg)

    def fromRouterMsg(self, msg: Message) -> Message:
        if isinstance(msg, TransferMessage):
            msg.from_node = self.node_mapping_inv[msg.from_node]
            msg.to_node = self.node_mapping_inv[msg.to_node]
            msg.inner_msg = self.fromRouterMsg(msg.inner_msg)
        elif isinstance(msg, DelayedMessage):
            msg.inner_msg = self.fromRouterMsg(msg.inner_msg)
        elif isinstance(msg, StateAnnouncementMsg):
            msg.node = self.node_mapping_inv[msg.node]

        return msg

    def toRouterMsg(self, msg: Message) -> Message:
        if isinstance(msg, TransferMessage):
            msg.from_node = self.node_mapping[msg.from_node]
            msg.to_node = self.node_mapping[msg.to_node]
            msg.inner_msg = self.fromRouterMsg(msg.inner_msg)
        elif isinstance(msg, DelayedMessage):
            msg.inner_msg = self.fromRouterMsg(msg.inner_msg)
        elif isinstance(msg, StateAnnouncementMsg):
            msg.node = self.node_mapping[msg.node]

        return msg
