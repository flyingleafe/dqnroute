import networkx as nx
import numpy as np
import itertools as it

from thespian.actors import *

from .router import AbstractRouter, RouterNotInitialized
from .router_mixins import RLAgent, ConveyorLinkStateHolder
from .messages import *
from .time_actor import *
from .utils import *

class Conveyor(AbstractRouter, ConveyorLinkStateHolder):
    def __init__(self):
        super().__init__()
        ConveyorLinkStateHolder.__init__(self)
        self.sections = []
        self.sinks = []
        self.speed = 0
        self.energy_consumption = 1
        self.stop_delay = 0
        self.energy_reward_weight = 0
        self.time_to_stop = 0
        self.stopping_event = None
        self.is_working = False
        self.sec_queue_times = {} # wow so hacky
        self.sec_process_time = 0
        self.neighbors_working = {}
        self.full_log = False

    def initialize(self, message, sender):
        my_id = super().initialize(message, sender)
        if isinstance(message, ConveyorInitMsg):
            self.sections = message.sections
            self.all_sections = message.all_sections
            self.sinks = message.sinks
            self.speed = message.speed
            self.energy_consumption = message.energy_consumption
            self.stop_delay = message.stop_delay
            self.energy_reward_weight = message.energy_reward_weight
            self.sec_process_time = message.sec_process_time
            self.full_log = message.full_log
            self.initGraph(self.all_sections, self.speed)
            for sec in self.sections.keys():
                self.sec_queue_times[sec] = 0
        elif isinstance(message, RouterFinalizeInitMsg):
            # self._announceLinkState()
            pass
        return my_id

    def _announceLinkState(self):
        announcement = self.mkLSAnnouncement(self.sections)
        self._broadcastAnnouncement(announcement, self.myAddress)
        self.seq_num += 1

    def _broadcastAnnouncement(self, announcement, sender):
        seen_targets = set()
        for n in self._allNeighbors():
            target = self.network[n]
            if sender != target and str(target) not in seen_targets:
                seen_targets.add(str(target))
                self.sendServiceMsg(target, announcement)

    def receiveServiceMsg(self, message, sender):
        super().receiveServiceMsg(message, sender)
        if isinstance(message, ConveyorLinkStateAnnouncement):
            if self.processLSAnnouncement(message, self.all_sections, self.speed):
                self._broadcastAnnouncement(message, sender)
        elif isinstance(message, BeltStatusMsg):
            for sec in message.sections:
                self.neighbors_working[sec] = message.working

    def _downstreamNeighbors(self):
        for sec_info in self.sections.values():
            up_nb = sec_info.get('upstream_neighbor', None)
            ad_nb = sec_info.get('adjacent_neighbor', None)
            if up_nb is not None and up_nb not in self.sections:
                yield up_nb
            if ad_nb is not None:
                yield ad_nb

    def _upstreamNeighbors(self):
        for (sec, sec_info) in self.all_sections.items():
            if sec in self.sinks:
                continue
            if sec in self.sections:
                continue
            up_nb = sec_info.get('upstream_neighbor', None)
            ad_nb = sec_info.get('adjacent_neighbor', None)
            if up_nb is not None and up_nb in self.sections:
                yield sec
            elif ad_nb is not None and ad_nb in self.sections:
                yield sec

    def _allNeighbors(self):
        return it.chain(self._downstreamNeighbors(), self._upstreamNeighbors())

    def _timeOnSection(self, sectionIdx):
        sec = self.sections[sectionIdx]
        return sec['length'] / self.speed

    def _delegateToNeighbor(self, n, bag_event):
        target = self.network[n]
        bag = bag_event.getContents()
        self.sendEvent(target, IncomingLuggageEvent(self.current_time, self.myAddress, n, bag_event.section, bag))

    def _scheduleSectionCrossing(self, bag_event):
        sec_num = bag_event.section
        bag = bag_event.getContents()

        bag.prev_time = self.current_time
        available_time = max(self.current_time, self.sec_queue_times[sec_num])
        self.sec_queue_times[sec_num] = available_time + self.sec_process_time

        finish_time = available_time + self._timeOnSection(sec_num)
        new_event = ProcessLuggageEvent(finish_time, bag_event.sender, sec_num, bag_event.prev_section, bag)
        self.event_queue.push(new_event)

    def _passToNextSection(self, bag_event):
        bag = bag_event.getContents()
        cur_section = bag_event.section
        up_neighbor = self.sections[bag_event.section]['upstream_neighbor']
        new_event = IncomingLuggageEvent(self.current_time, self.myAddress, up_neighbor, cur_section, bag)
        self.recordOutgoingBag(bag, cur_section)
        if up_neighbor not in self.sections:
            # last section, pass to upstream
            target = self.network[up_neighbor]
            self.sendEvent(target, new_event)
        else:
            self.event_queue.push(new_event)

    def _scheduleStopping(self, time):
        if time > self.time_to_stop:
            self.time_to_stop = time
            if self.stopping_event is None:
                self.stopping_event = StopMovingEvent(time)
                self.event_queue.push(self.stopping_event)
            else:
                self.event_queue.change_time(self.stopping_event, time)

    def _setWorkStatus(self, is_working):
        self.is_working = is_working
        for nb in self._upstreamNeighbors():
            self.sendServiceMsg(self.network[nb], BeltStatusMsg(list(self.sections.keys()), self.is_working))

    # def isInitialized(self):
    #     return super().isInitialized() and (len(self.announcements) == len(self.all_sections))

    def processEvent(self, event):
        if not self.isInitialized():
            raise RouterNotInitialized("Conveyor has not been initialized!")
        if isinstance(event, IncomingLuggageEvent):
            bag = event.getContents()
            if bag.dst == event.section:
                self.reportBagDone(bag, self.current_time)
            else:
                if not self.is_working:
                    self._setWorkStatus(True)
                self._scheduleSectionCrossing(event)
        elif isinstance(event, ProcessLuggageEvent):
            event.prev_belt_stop_time = self.time_to_stop
            self._scheduleStopping(self.current_time + self.stop_delay)

            self.reportBagDelegation(event)
            section = event.section
            bag = event.getContents()
            bag.energy_spent += self._computeEnergyOverhead(event)

            if self.full_log:
                bag.route_add(self._currentStateData(pkg, section), self._currentStateCols())
            else:
                bag.route_add([self.current_time, section], ['time', 'cur_node'])

            print("CONVEYOR {} (SECTION {}) ROUTES BAG {} TO SINK {}".format(self.addr, section, bag.id, bag.dst))
            section_info = self.sections[section]
            if section_info.get('adjacent_neighbor', None) is None:
                self._passToNextSection(event)
            else:
                next_section = self.routeBagToPossiblePath(bag, section)
                if section_info['upstream_neighbor'] == next_section:
                    self._passToNextSection(event)
                elif next_section == section_info['adjacent_neighbor']:
                    self._delegateToNeighbor(next_section, event)
                else:
                    raise Exception("Trying to route to non-adjacent section!")
        elif isinstance(event, StopMovingEvent):
            self.stopping_event = None
            if self.is_working:
                self._setWorkStatus(False)

    def reportBagDone(self, bag, time):
        self.send(self.overlord, BagDoneMsg(time, self.myAddress, bag))

    def _possibleNeighbors(self, section, dst):
        res = []
        for nb in self.network_graph.neighbors(section):
            if nx.has_path(self.network_graph, nb, dst):
                res.append(nb)
        return res

    def routeBagToPossiblePath(self, bag, section):
        d = bag.dst
        pnbs = self._possibleNeighbors(section, d)
        if len(pnbs) == 0:
            raise Exception('Bag {} in section {} cannot reach its destination {} at all!'
                            .format(bag.id, section, d))
        elif len(pnbs) == 1:
            self.recordOutgoingBag(bag, section)
            return pnbs[0]
        else:
            return self.routeBag(bag, section)

    def routeBag(self, bag, section):
        pass

    def reportBagDelegation(self, bag_event):
        pass

    def recordOutgoingBag(self, bag, section):
        pass

    def _neighborsWorkingEnc(self):
        enc = np.zeros(len(self.all_sections))
        for (nb, working) in self.neighbors_working.items():
            enc[nb] = 1 if working else 0
        return enc

    def _computeEnergyOverhead(self, bag_event):
        new_stop_time = bag_event.time + self.stop_delay
        time_to_overwork = max(0, new_stop_time - max(bag_event.prev_belt_stop_time, bag_event.time))
        return self.energy_consumption * time_to_overwork

    def _computeTimeOverhead(self, bag_event):
        bag = bag_event.getContents()
        return bag_event.time - bag.prev_time

    def _computeReward(self, bag_event):
        time_passed = self._computeTimeOverhead(bag_event)
        energy_required = self._computeEnergyOverhead(bag_event)
        return -(time_passed + self.energy_reward_weight * energy_required)

    def _currentStateData(self, pkg, section):
        return mk_current_neural_state(self.network_graph, self.current_time, pkg, section,
                                       self._neighborsWorkingEnc())

    def _currentStateCols(self):
        return get_data_cols(len(self.all_sections))

class LinkStateConveyor(Conveyor):
    # def initialize(self, message, sender):
    #     my_id = super().initialize(message, sender)
    #     if isinstance(message, LinkStateConveyorInitMsg):
    #         self._announceLinkState()
    #     return my_id

    def routeBag(self, bag, section):
        d = bag.dst
        path = nx.dijkstra_path(self.network_graph, section, d)
        return path[1]

class QConveyor(Conveyor, RLAgent):
    def __init__(self):
        super().__init__()
        self.reward_pending = {}

    def reportBagDelegation(self, bag_event):
        prev_sec = bag_event.prev_section
        reward_msg = self.mkRewardMsg(bag_event)
        if prev_sec in self.sections:
            self.receiveServiceMsg(reward_msg, self.myAddress)
        elif prev_sec != -1:
            self.sendServiceMsg(self.network[prev_sec], reward_msg)

    def recordOutgoingBag(self, bag, section):
        state = self.getState(bag, section)
        self.reward_pending[bag.id] = state
        return state

    def routeBag(self, bag, section):
        state = self.recordOutgoingBag(bag, section)
        return self.act(state, bag)

    def receiveServiceMsg(self, message, sender):
        super().receiveServiceMsg(message, sender)
        if isinstance(message, ConveyorRewardMsg):
            prev_state = self.reward_pending[message.bag_id]
            del self.reward_pending[message.bag_id]
            self.observe(self.mkSample(message, prev_state, sender))

    def mkRewardMsg(self, bag_event):
        pass

    def getState(self, bag, section):
        pass

    def mkSample(self, message, prev_state, sender):
        pass

class SimpleQConveyor(QConveyor):
    def __init__(self):
        super().__init__()
        self.Q = {}
        self.learning_rate = None
        self.broken_links = {}
        self.broken_link_Qs = {}

    def initialize(self, message, sender):
        my_id = super().initialize(message, sender)
        if isinstance(message, SimpleQConveyorInitMsg):
            self.learning_rate = message.learning_rate
            for (sec_num, sec_info) in self.sections.items():
                self.Q[sec_num] = {}
                self.broken_links[sec_num] = {}
                self.broken_link_Qs[sec_num] = {}
                nbs = sec_neighbors_list(sec_info)
                for s in self.sinks:
                    self.Q[sec_num][s] = {}
                    for nb in nbs:
                        if s == nb:
                            self.Q[sec_num][s][nb] = 0
                        else:
                            self.Q[sec_num][s][nb] = 1
            # self._announceLinkState()
        return my_id

    def _mkBestEstimate(self, sec_num, d):
        result_q = {}
        for n in self._possibleNeighbors(sec_num, d):
            if n not in self.broken_links[sec_num]:
                result_q[n] = self.Q[sec_num][d][n]
        return dict_min(result_q)

    def getState(self, bag, section):
        return (self.current_time, section, bag.dst)

    def mkRewardMsg(self, bag_event):
        bag = bag_event.getContents()
        d = bag.dst
        section = bag_event.section
        reward = -self._computeReward(bag_event)
        best_estimate = 0 if section == d else self._mkBestEstimate(section, d)[1]
        return SimpleQConveyorRewardMsg(bag.id, section, reward, best_estimate)

    def mkSample(self, message, prev_state, sender):
        if isinstance(message, SimpleQConveyorRewardMsg):
            a = message.section
            s = prev_state[1:]
            r = message.reward
            q = message.estimate
            return (s, a, r, q)
        else:
            raise Exception("Unsupported type of reward msg!")

    def act(self, state, bag):
        _, sec_num, d = state
        return self._mkBestEstimate(sec_num, d)[0]

    def observe(self, sample):
        ((sec_num, d), sender_sec, reward, estimate) = sample
        delta = self.learning_rate * (reward + estimate - self.Q[sec_num][d][sender_sec])
        self.Q[sec_num][d][sender_sec] += delta
