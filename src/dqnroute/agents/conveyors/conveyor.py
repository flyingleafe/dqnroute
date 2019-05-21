import pprint
import networkx as nx

from typing import List, Tuple
from .common import *
from ..base import *
from ...messages import *
from ...conveyor_model import *
from ...utils import *

CONV_SUB_MODEL_LEN = 3

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

    # def log(self, msg, force=False):
    #     super().log(msg, force or self.id[1] == 6)

    def handleBagMsg(self, sender: AgentId, msg: ConveyorBagMsg) -> List[WorldEvent]:
        evs = self._interruptMovement()
        bag = msg.bag

        if isinstance(msg, IncomingBagMsg):
            n_node, n_pos = self.notifierPos(sender)
            self.log('bag #{} arrives from {} at {}, pos {}'
                     .format(bag.id, sender, n_node, n_pos))

            initially_empty = len(self.model.objects) == 0
            self.model.putObject(bag.id, bag, n_pos)
            self.bag_checkpoints[bag.id] = set()
            evs += self.handleIncomingBag(n_node, bag)
            if initially_empty:
                evs += self._announceState()

        elif isinstance(msg, OutgoingBagMsg):
            assert agent_type(sender) == 'diverter', "only diverters send us such stuff!"

            self.log('bag #{} leaves at {}, pos {}'
                     .format(bag.id, sender, self.model.checkpointPos(sender)))
            self.model.removeObject(bag.id)
            self.bag_checkpoints.pop(bag.id)
            evs += self.handleOutgoingBag(sender, bag)
            if len(self.model.objects) == 0:
                evs += self._announceState()

        elif isinstance(msg, PassedBagMsg):
            assert agent_type(sender) == 'diverter', "only diverters send us such stuff!"
            self.log('bag #{} passes at {}, pos {}'
                     .format(bag.id, sender, self.model.checkpointPos(sender)))
            dv_pos = self.model.checkpointPos(sender)
            o_pos = self.model.objPos(bag.id)
            if dv_pos != o_pos:
                self.log('DIVERGENCE WITH REALITY: {} passed the {} in {}m, while we thought it is in {}m'
                         .format(bag, sender, dv_pos, o_pos), True)

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
            self.log('new conv speed: {}'.format(new_speed))
            return [ConveyorSpeedChangeAction(new_speed)] + self._announceState()
        else:
            return []

    def getState(self):
        return {'speed': self.model.speed, 'empty': len(self.model.objects) == 0}

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

                self.log('handling eveng: {} {} {}'.format(bag, node, delay))

                if node not in self.bag_checkpoints[bag.id]:
                    if agent_type(node) == 'conv_end':
                        self.model.removeObject(bag.id)
                        evs += self.handleOutgoingBag(node, bag)
                    else:
                        evs += self.checkpointReach(node, bag)
                        self.bag_checkpoints[bag.id].add(node)
                steps += 1

            evs += self.beforeResume()
            self.model.endResolving()
        else:
            evs += self.beforeResume()
            if self.model.dirty():
                self.model.startResolving()
                self.model.endResolving()

        return evs + self._resumeMovement()

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

        self.log('MOVING ({}m/s): {} {} {}'.format(self.model.speed, conv_events,
                                                   self.model.object_positions, self.model.checkpoints))

        evs = []
        if len(conv_events) > 0:
            self.conv_movement_start_time = self.env.time()
            delay = conv_events[0][2]
            ev = self.delayed(delay, self._pauseResolve)
            self.delayed_conv_update = ev.id
            evs += [ev]
        else:
            evs += self._pause()
        return evs

    def _pause(self):
        self.model.pause(self.env.time())
        return self.onPause()

    def _interruptMovement(self) -> List[WorldEvent]:
        evs = []
        if self.hasDelayed(self.delayed_conv_update):
            evs += self._pause() + [self.cancelDelayed(self.delayed_conv_update)]
        return evs

    def _pauseResolve(self) -> List[WorldEvent]:
        evs = self._pause()
        return evs + self._resolveAndResume()

    def beforeResume(self):
        return []

    def onPause(self):
        return []

class StopDelayConveyor(BaseConveyor):
    """
    Conveyor which stops after some time has passed
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


class SimpleRouterConveyor(StopDelayConveyor, RouterContainer):
    """
    Simple conveyor which also handles the logic of virtual junction routers.
    """
    def __init__(self, id: AgentId, topology: nx.DiGraph, **kwargs):
        checkpoints = conveyor_adj_nodes(topology, agent_idx(id), only_own=True,
                                         data='conveyor_pos')
        super().__init__(id=id, checkpoints=checkpoints,
                         topology=topology, **kwargs)

        conv_idx = agent_idx(self.id)
        self.conv_edges = conveyor_edges(self.topology, conv_idx)

        self.next_node = {}
        self.up_conv = {}
        self.down_conv = {}
        self.up_conv_inv = {}
        self.down_conv_inv = {}
        for i, (u, v) in enumerate(self.conv_edges):
            self.next_node[u] = v
            if agent_type(u) == 'junction':
                cid = [cid for _, _, cid in self.topology.in_edges(u, data='conveyor')
                       if cid != conv_idx][0]
                self.down_conv[u] = cid
                self.down_conv_inv[cid] = u
            elif i > 0 and agent_type(u) == 'diverter':
                cid = [cid for _, _, cid in self.topology.out_edges(u, data='conveyor')
                       if cid != conv_idx][0]
                self.up_conv[u] = (cid, 0)
                self.up_conv_inv[cid] = u

        self.downstream_node = self.conv_edges[0][0]
        self.upstream_node = self.conv_edges[-1][1]
        up_ps = self.topology.nodes[self.upstream_node]
        up_ps = (up_ps.get('conveyor', -1), up_ps.get('conveyor_pos', 0))
        self.up_conv[self.upstream_node] = up_ps
        self.up_conv_inv[up_ps[0]] = self.upstream_node

        self.up_model = {}
        for (cid, pos) in self.up_conv.values():
            if cid != -1:
                self.up_model[cid] = ConveyorModel(
                    CONV_SUB_MODEL_LEN + pos, self.model.max_speed, [],
                    ('conv_sub_{}'.format(self.id[1]), cid))

        self._prev_junc_nodes = {}
        self.dv_kick_estimate = {}

    def handleOutgoingBag(self, node: AgentId, bag: Bag) -> List[WorldEvent]:
        evs = super().handleOutgoingBag(node, bag)
        if agent_type(node) == 'conv_end':
            node = self.upstream_node

        up_conv, up_pos = self.up_conv[node]
        if up_conv != -1:
            self.log('bag #{} leaves from {} to upstream conv {}'
                     .format(bag.id, node, up_conv))
            self.up_model[up_conv].putObject(bag.id, bag, up_pos)

            if agent_type(node) != 'diverter':
                evs += [OutMessage(self.id, ('conveyor', up_conv), IncomingBagMsg(bag))]
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
            arrival_node = self.down_conv_inv[notifier[1]]
            return arrival_node, self.model.checkpointPos(arrival_node)
        else:
            raise Exception('invalid notifier: {}'.format(notifier))

    def processNewAnnouncement(self, node: AgentId, state) -> Tuple[bool, List[WorldEvent]]:
        res, msgs = super().processNewAnnouncement(node, state)
        if res:
            if agent_type(node) == 'conveyor':
                cid = agent_idx(node)
                if cid in self.up_conv_inv and not state['empty']:
                    msgs += self._interruptMovement()
                    self.up_model[cid].setSpeed(state['speed'])
                    self.log('speed change announcement! conv {} -> {}m/s'.format(cid, state['speed']))

                    msgs += self._resolveAndResume()
        return res, msgs

    def onPause(self):
        for model in self.up_model.values():
            model.pause(self.env.time())
        return []

    def beforeResume(self):
        msgs = self._cascadeUpdate()
        for model in self.up_model.values():
            if model.dirty():
                model.startResolving()
                model.endResolving()
            model.resume(self.env.time())
        return []

    def _cascadeUpdate(self):
        if len(self.model.objects) == 0:
            return []

        max_speed = self.model.max_speed
        for u, v in reversed(self.conv_edges):
            up_conv, up_pos = self.up_conv.get(v, (-1, 0))
            if up_conv == -1:
                continue

            v_pos = self.model.checkpointPos(v) or self.model.length
            u_pos = self.model.checkpointPos(u) or 0
            nxt_exiting = self.model.nearestObject(v_pos, preference='prev')
            if nxt_exiting is None:
                continue

            bag, bag_pos = nxt_exiting
            dist_to_end = v_pos - bag_pos
            assert dist_to_end >= 0, \
                "searched for objs prev {}, found ({}, {})! ({} {})".format(v_pos, bag, bag_pos, u, v)

            up_model = self.up_model[up_conv]
            self.log('model of conv {}: {}m/s, {}'.format(up_conv, up_model.speed, up_model.object_positions))
            if agent_type(v) == 'diverter':
                if self.dv_kick_estimate.get(bag.id, False):
                    max_speed = min(max_speed, find_max_speed(up_model, up_pos, up_model.speed,
                                                          dist_to_end, max_speed))
            else:
                max_speed = min(max_speed, find_max_speed(up_model, up_pos, up_model.speed,
                                                          dist_to_end, max_speed))

        if max_speed != self.model.speed:
            self.log('CASCADE SPEED CHANGE: {}m/s'.format(max_speed))
        return self.setSpeed(max_speed)
