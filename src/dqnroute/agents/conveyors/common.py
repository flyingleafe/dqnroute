import pprint
import networkx as nx

from typing import List, Tuple, Optional
from copy import deepcopy
from ..base import *
from ..routers import *
from ...messages import *
from ...utils import *


class ConveyorStateHandler(AbstractStateHandler):
    """
    Exchanges info about state of controllers in the system,
    such as speed of conveyors.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.controllers_state = {}

    def processNewAnnouncement(self, node: AgentId, state) -> bool:
        try:
            old_state = self.controllers_state[node]
            if old_state == state:
                return False
        except KeyError:
            pass
        self.controllers_state[node] = state
        return True


class SimpleSource(BagDetector, ConveyorStateHandler):
    """
    Class which implements a bag source controller, which notifies
    the system about a new bag arrival.
    """
    def bagDetection(self, bag: Bag) -> List[WorldEvent]:
        nbr = self.interface_map[0]    # source is always connected only to upstream conv
        return [OutMessage(self.id, nbr, IncomingBagMsg(bag))]

    def getState(self):
        return None

class SimpleSink(BagDetector, ConveyorStateHandler):
    """
    Class which implements a sink controller, which detects
    an exit of a bag from the system.
    """
    def bagDetection(self, bag: Bag) -> List[WorldEvent]:
        return [BagReceiveAction(bag)]

    def getState(self):
        return None


class RouterContainer(MessageHandler):
    """
    Class which adapts router logic in such a way that routers
    think that they are working in computer network isomorphic to the
    topology graph of conveyor network
    """
    def __init__(self, topology: nx.DiGraph, router_factory, **kwargs):
        super().__init__(**kwargs)

        self.topology = topology
        self.node_mapping = {}
        self.node_mapping_inv = {}
        for (i, aid) in enumerate(sorted(self.topology.nodes)):
            rid = ('router', i)
            self.node_mapping[aid] = rid
            self.node_mapping_inv[rid] = aid

        self.virt_conn_graph = nx.relabel_nodes(self.topology, self.node_mapping).to_undirected()
        self.routers = {}

        self.env.register('conn_graph', self.virt_conn_graph)
        router_factory._setEnv(self.env)

        for rid in self.childrenRouters(self.id):
            self.routers[rid] = router_factory._makeHandler(rid)

    def init(self, config) -> List[WorldEvent]:
        msgs = super().init(config)
        init_msg = WireInMsg(-1, InitMessage(config))
        return msgs + flatten([self.handleViaRouter(rid, init_msg) for rid in self.routers.keys()])

    def handleMsgFrom(self, sender: AgentId, msg: Message) -> List[WorldEvent]:
        if isinstance(msg, WrappedRouterMsg):
            to_router = msg.to_router
            if to_router in self.routers:
                return self.handleViaRouter(to_router, msg.inner)
            else:
                return [self.sendWrapped(msg)]
        else:
            return super().handleMsgFrom(sender, msg)

    def parentCtrl(self, router_id: AgentId) -> AgentId:
        """
        Returns an ID of a controller in the system which is
        responsible for simulating the behavior of a router with
        a given ID.
        """
        top_node = self.node_mapping_inv[router_id]
        if agent_type(top_node) == 'junction':
            conv_idx = self.topology.nodes[top_node]['conveyor']
            return ('conveyor', conv_idx)
        else:
            return top_node

    def childrenRouters(self, agent_id: AgentId) -> List[AgentId]:
        """
        Returns a list of router nodes for which a given controller
        is responsible
        """
        if agent_type(agent_id) == 'conveyor':
            conv_idx = agent_idx(agent_id)
            return [
                self.node_mapping[n]
                for n in conveyor_adj_nodes(self.topology, conv_idx, only_own=True)
                if agent_type(n) == 'junction'
            ]
        else:
            return [self.node_mapping[agent_id]]

    def sendWrapped(self, msg: WrappedRouterMsg) -> Message:
        """
        Transforms a given message so that it is sent to:
        - loopback, if target router is our own
        - to a parent controller directly, if it is among our neighbours
        - to a neighbor which should be the closest to the parent controller
          accordingly to system topology, otherwise
        """
        from_router = msg.from_router
        to_router = msg.to_router
        to_ctrl = self.parentCtrl(to_router)

        if to_ctrl != self.id:
            if to_ctrl in self.interface_inv_map:
                return OutMessage(self.id, to_ctrl, msg)
            else:
                # this can only be in case when the next topgraph node
                # is separated from us via one-section conveyor
                from_node = self.node_mapping_inv[from_router]
                to_node = self.node_mapping_inv[to_router]
                try:
                    middle_conv = self.topology[from_node][to_node]['conveyor']
                except KeyError:
                    middle_conv = self.topology[to_node][from_node]['conveyor']
                return OutMessage(self.id, ('conveyor', middle_conv), msg)
        else:
            return msg

    def fromRouterEvent(self, router_id: AgentId, event: WorldEvent) -> List[WorldEvent]:
        """
        Transform a router event into possibly zero or several
        events in conveyor network
        """
        assert router_id in self.routers, "Cannot process an event related to not own router!"

        if isinstance(event, DelayedEvent):
            return [DelayedEvent(event.id, event.delay,
                                 self.fromRouterEvent(router_id, event.inner))]

        elif isinstance(event, DelayInterrupt):
            return [event]

        elif isinstance(event, Action):
            return self.fromRouterAction(router_id, event)

        elif isinstance(event, Message):
            if isinstance(event, WireOutMsg):
                int_id = event.interface
                inner = event.payload
                to_router, to_interface = resolve_interface(self.virt_conn_graph, router_id, int_id)
                new_msg = WrappedRouterMsg(router_id, to_router, WireInMsg(to_interface, inner))
                return [self.sendWrapped(new_msg)]
            else:
                raise UnsupportedMessageType(event)

        else:
            raise UnsupportedEventType(event)

    def bagToPkg(self, bag: Bag) -> Bag:
        """
        Changes the bag destination to router rep
        """
        new_bag = deepcopy(bag)
        new_bag.dst = self.node_mapping[bag.dst]
        return new_bag

    def pkgToBag(self, bag: Bag) -> Bag:
        new_bag = deepcopy(bag)
        new_bag.dst = self.node_mapping_inv[bag.dst]
        return new_bag

    def fromRouterAction(self, router_id: AgentId, action: Action) -> List[WorldEvent]:
        """
        React on some action performed by underlying router.
        Do nothing by default. Subclasses should override this.
        """
        return []

    def handleViaRouter(self, router_id: AgentId, event: WorldEvent) -> List[WorldEvent]:
        """
        Passes an event to a given router and transforms events spawned by router
        """
        router_evs = self.routers[router_id].handle(event)
        return flatten([self.fromRouterEvent(router_id, rev) for rev in router_evs])

    def handleBagViaRouter(self, from_router: AgentId, router_id: AgentId, bag: Bag) -> List[WorldEvent]:
        """
        Enqueues and processes a `Bag` as `Package` in the same time.
        """
        node_id = self.node_mapping_inv[router_id]
        allowed_nbrs = only_reachable(self.topology, bag.dst, self.topology.successors(node_id))
        allowed_routers = [self.node_mapping[v] for v in allowed_nbrs]

        pkg = self.bagToPkg(bag)
        enqueued_evs = self.handleViaRouter(router_id, PkgEnqueuedEvent(from_router, router_id, pkg))
        process_evs = self.handleViaRouter(router_id,
                                           PkgProcessingEvent(from_router, router_id, pkg, allowed_routers))
        return enqueued_evs + process_evs

    def routerId(self):
        """
        Returns the ID of the corresponding router if there is only one
        such router, throws an exception otherwise.
        """
        if len(self.routers) == 1:
            return self.node_mapping[self.id]
        else:
            raise Exception('Agent {} has more than 1 virtual router'.format(self.id))


class RouterSource(RouterContainer, SimpleSource):
    """
    Source controller which contains a virtual router. Does the
    same as `SimpleSource` but also manages virtual router messages
    on behalf of its node.
    """
    def bagDetection(self, bag: Bag) -> List[WorldEvent]:
        sender = ('world', 0)
        router_id = self.routerId()
        nbr = self.interface_map[0] # source is always connected only to upstream conv
        return [OutMessage(self.id, nbr, IncomingBagMsg(bag))] + self.handleBagViaRouter(sender, router_id, bag)


class RouterSink(RouterContainer, SimpleSink):
    """
    Sink controller which contains a virtual router. Does the same
    as `SimpleSource` but also manages virtual router messages.
    """
    def bagDetection(self, bag: Bag) -> List[WorldEvent]:
        router_id = self.routerId()
        last_conv = ('conveyor', bag.last_conveyor)

        sender = None
        for v, _, ps in self.topology.in_edges(self.id, data=True):
            conv_idx = ps['conveyor']
            if ('conveyor', conv_idx) == last_conv:
                sender = self.node_mapping[v]
                break
        if sender is None:
            raise Exception('Prev node on source conveyor is not found!')

        return self.handleBagViaRouter(sender, router_id, bag)

    def fromRouterAction(self, router_id: AgentId, action: Action) -> List[WorldEvent]:
        if isinstance(action, PkgReceiveAction):
            return [BagReceiveAction(self.pkgToBag(action.pkg))]
        else:
            return super().fromRouterAction(router_id, action)

