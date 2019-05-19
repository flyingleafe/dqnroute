import pprint
import networkx as nx

from typing import List, Tuple
from .common import *
from ..base import *
from ...messages import *
from ...conveyor_model import *
from ...utils import *


class BaseConveyor(Conveyor, ConveyorStateHandler):
    """
    Conveyor which can handle its length and speed and can track the positions of bags.
    Should be only used as a superclass.
    """
    def __init__(self, length: float, max_speed: float,
                 checkpoints: List[Tuple[AgentId, float]], **kwargs):
        super().__init__(**kwargs)
        self.model = ConveyorModel(length, max_speed, checkpoints, model_id=self.id)
        self.delayed_conv_update = -1
        self.bag_checkpoints = {}

    def handleBagMsg(self, sender: AgentId, msg: ConveyorBagMsg) -> List[WorldEvent]:
        evs = self._interruptMovement()
        bag = msg.bag

        if isinstance(msg, IncomingBagMsg):
            n_node, n_pos = self.notifierPos(sender)
            self.log('bag #{} arrives from {} at {}, pos {}'
                     .format(bag.id, sender, n_node, n_pos))

            self.model.putObject(bag.id, bag, n_pos)
            self.bag_checkpoints[bag.id] = set()
            evs += self.handleIncomingBag(n_node, bag)

        elif isinstance(msg, OutgoingBagMsg):
            assert agent_type(sender) == 'diverter', "only diverters send us such stuff!"

            self.log('bag #{} leaves at {}, pos {}'
                     .format(bag.id, sender, self.model.checkpointPos(sender)))
            self.model.removeObject(bag.id)
            self.bag_checkpoints.pop(bag.id)
            evs += self.handleOutgoingBag(sender, bag)

        elif isinstance(msg, PassedBagMsg):
            assert agent_type(sender) == 'diverter', "only diverters send us such stuff!"
            self.log('bag #{} passes at {}, pos {}'
                     .format(bag.id, sender, self.model.checkpointPos(sender)))
            evs += self.handlePassedBag(sender, bag)

        return evs + self._resolveAndResume()

    def notifierPos(self, agent: AgentId) -> Tuple[float]:
        raise NotImplementedError()

    def handleIncomingBag(self, notifier: AgentId, bag: Bag) -> List[WorldEvent]:
        raise NotImplementedError()

    def handleOutgoingBag(self, notifier: AgentId, bag: Bag) -> List[WorldEvent]:
        raise NotImplementedError()

    def handlePassedBag(self, notifier: AgentId, bag: Bag) -> List[WorldEvent]:
        raise NotImplementedError()

    def checkpointReach(self, node: AgentId, bag: Bag) -> List[WorldEvent]:
        raise NotImplementedError()

    def start(self, speed=None) -> List[WorldEvent]:
        if speed is None:
            speed = self.model.max_speed
        return self.setSpeed(speed)

    def stop(self) -> List[WorldEvent]:
        return self.setSpeed(0)

    def setSpeed(self, new_speed: float) -> List[WorldEvent]:
        if self.model.speed != new_speed:
            self.model.setSpeed(new_speed)
            return [ConveyorSpeedChangeAction(new_speed)] + self._announceState()
        else:
            return []

    def getState(self):
        return {'speed': self.model.speed}

    def _resolveAndResume(self) -> List[WorldEvent]:
        evs = []
        if self.model.dirty():
            self.model.startResolving()
            steps = 0
            for bag, node, delay in self._allUnresolvedEvents():
                if steps > 100:
                    self.log('wtf?\n  {}\n  {}\n  {}\n  {}\n'
                             .format(ev, self.model.checkpoints,
                                     self.model.object_positions, self.model._resolved_events))
                    raise Exception('okay what the fuck?')

                if node not in self.bag_checkpoints[bag.id]:
                    if agent_type(node) == 'conv_end':
                        self.model.removeObject(bag.id)
                        evs += self.handleOutgoingBag(self.id, bag)
                    else:
                        evs += self.checkpointReach(node, bag)
                        self.bag_checkpoints[bag.id].add(node)
                steps += 1

            self.model.endResolving()

        if not self.model.resolving():
            evs += self._resumeMovement()
        return evs

    def _allUnresolvedEvents(self):
        while True:
            ev = self.model.pickUnresolvedEvent()
            if ev is None:
                break
            yield ev

    def _resumeMovement(self) -> List[WorldEvent]:
        if self.hasDelayed(self.delayed_conv_update):
            raise Exception('why it still has delayed?')

        self.model.resume(self.env.time())
        conv_events = self.model.nextEvents()

        if len(conv_events) > 0:
            self.conv_movement_start_time = self.env.time()
            delay = conv_events[0][2]
            ev = self.delayed(delay, self._pauseResolve)
            self.delayed_conv_update = ev.id
            return [ev]
        else:
            self.model.pause(self.env.time())
            return []

    def _interruptMovement(self) -> List[WorldEvent]:
        if self.hasDelayed(self.delayed_conv_update):
            self.model.pause(self.env.time())
            return [self.cancelDelayed(self.delayed_conv_update)]
        return []

    def _pauseResolve(self) -> List[WorldEvent]:
        self.model.pause(self.env.time())
        return self._resolveAndResume()


class SimpleConveyor(BaseConveyor):
    """
    Simple conveyor which stops after some time has passed
    since the last bag exited it. It is connected to routers
    (diverters) seated on it and notifies them when it starts or stops.
    """
    def __init__(self, stop_delay: float, **kwargs):
        super().__init__(**kwargs)
        self.stop_delay = stop_delay
        self.delayed_stop = -1

    def handleIncomingBag(self, notifier: AgentId, bag: Bag) -> List[WorldEvent]:
        if self.model.speed == 0:
            return self.start()
        else:
            return self._cancelDelayedStop()

    def handleOutgoingBag(self, notifier: AgentId, bag: Bag) -> List[WorldEvent]:
        if len(self.model.objects) == 0:
            ev = self.delayed(self.stop_delay, self.stop)
            self.delayed_stop = ev.id
            # if self.delayed_stop > 2000 and self.delayed_stop % 100 == 0:
            #     print('{}: super large delayed stops! ({})'.format(self.id, self.delayed_stop))
            return [ev]
        return []

    def handlePassedBag(self, notifier: AgentId, bag: Bag) -> List[WorldEvent]:
        return []

    def _cancelDelayedStop(self) -> List[WorldEvent]:
        if self.hasDelayed(self.delayed_stop):
            return [self.cancelDelayed(self.delayed_stop)]
        return []


class SimpleRouterConveyor(SimpleConveyor, RouterContainer):
    """
    Simple conveyor which also handles the logic of virtual junction routers.
    """
    def __init__(self, id: AgentId, topology: nx.DiGraph, **kwargs):
        checkpoints = conveyor_adj_nodes(topology, agent_idx(id), only_own=True,
                                         data='conveyor_pos')
        super().__init__(id=id, checkpoints=checkpoints,
                         topology=topology, **kwargs)

        self.prev_convs_ups = {}
        for node, _ in self.model.checkpoints:
            prev = prev_adj_conv_node(self.topology, node)
            if prev is not None:
                _, cid = prev
                self.prev_convs_ups[('conveyor', cid)] = node

        conv_edges = conveyor_edges(self.topology, self.id[1])
        self.downstream_node = conv_edges[0][0]
        self.upstream_node = conv_edges[-1][1]
        self._prev_junc_nodes = {}

    def handleOutgoingBag(self, node: AgentId, bag: Bag) -> List[WorldEvent]:
        evs = super().handleOutgoingBag(node, bag)
        if node == self.id:
            if agent_type(self.upstream_node) != 'sink':
                up_conv = ('conveyor', conveyor_idx(self.topology, self.upstream_node))
                self.log('bag #{} leaves from {} to upstream conv {}'
                         .format(bag.id, node, up_conv))

                evs += [OutMessage(self.id, up_conv, IncomingBagMsg(bag))]
        return evs

    def handleIncomingBag(self, node: AgentId, bag: Bag) -> List[WorldEvent]:
        evs = super().handleIncomingBag(node, bag)
        if agent_type(node) == 'junction':
            prev_node, _ = prev_adj_conv_node(self.topology, node)
            self._prev_junc_nodes[bag.id] = prev_node
            self.log('remembered prev node {} for bag #{} arrived on {}'
                     .format(prev_node, bag.id, node))
        return evs

    def handlePassedBag(self, node: AgentId, bag: Bag) -> List[WorldEvent]:
        return []

    def checkpointReach(self, node: AgentId, bag: Bag) -> List[WorldEvent]:
        if agent_type(node) == 'junction':
            try:
                prev_node = self._prev_junc_nodes.pop(bag.id)
            except KeyError:
                prev_node = prev_same_conv_node(self.topology, node)
                if prev_node is None:    # junction in the beginning
                    prev_node, _ = prev_adj_conv_node(self.topology, node)

            self.log('bag #{} on checkpoint {} ({}m, prev: {})'
                     .format(bag.id, node, self.model.checkpointPos(node), prev_node))

            sender = self.node_mapping[prev_node]
            router_id = self.node_mapping[node]

            # if bag.id == 638:
            #     print('HEY IT WAS HERE: {} - {}'.format(node, router_id))
            return self.handleBagViaRouter(sender, router_id, bag)
        else:
            return []

    def notifierPos(self, notifier: AgentId) -> Tuple[AgentId, float]:
        n_type = agent_type(notifier)
        if n_type in ('source', 'diverter'):
            return notifier, 0
        elif n_type == 'conveyor':
            arrival_node = self.prev_convs_ups[notifier]
            assert agent_type(arrival_node) == 'junction', "hey wats up!!"
            return arrival_node, self.model.checkpointPos(arrival_node)
        else:
            raise Exception('invalid notifier: {}'.format(notifier))
