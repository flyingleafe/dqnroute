import random
import math
import numpy as np
import networkx as nx
import tensorflow as tf
import pandas as pd

from collections import deque

from .messages import *
from .router import *
from .memory import *
from .utils import *
from .router_mixins import LinkStateHolder
from .networks import get_qnetwork_class

MAX_EPSILON = 1
MIN_EPSILON = 0.01
LAMBDA = 0.01

MAX_TEMP = 10.0
MIN_TEMP = 1.5
DECAY_TEMP_STEPS = 60000

LOAD_LVL_WEIGHTS = [10, 8, 6, 5, 4, 3, 2, 2, 1, 1]
LOAD_LVL_LOW_THRESHOLD = 1.0
LOAD_LVL_HIGH_THRESHOLD = 2.0

class DQNRouter(QRouter, LinkStateHolder):
    def __init__(self):
        super().__init__()
        LinkStateHolder.__init__(self)
        self.batch_size = 1
        self.pkg_states = False
        self.brain = None
        self.memory = None
        self.prioritized_xp = False
        self.temp = MIN_TEMP
        self.steps = 0
        self.err_mavg = None
        self.session = None
        self.outgoing_pkgs_num = {}
        self.load_lvl_mavg = deque([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 10)
        self.neighbors_advices = {}
        self.neighbors_load_statuses = {}
        self.ploho_flag = False
        self.recent_rnn_state = None

    def _initModel(self, n, nn_type, settings):
        tf.reset_default_graph()
        init = tf.global_variables_initializer()
        NetworkClass = get_qnetwork_class(nn_type)
        self.brain = NetworkClass(n, **settings)
        self.session = tf.Session()
        # load model
        self.session.run(init)
        self.brain.restore(self.session)
        print('Restored model from ' + self.brain.getSavePath())
        # self.temp = MAX_TEMP
        # print('No pre-trained model loaded')

    def _train(self, x, y, batch_size=1, outer_state=None, save_old_state=False):
        if not self.pkg_states:
            outer_state = None
        self.brain.fit(self.session, x, y, batch_size=batch_size, outer_state=outer_state,
                       save_old_state=save_old_state)

    def _predict(self, x, rstate=None, batch_size=1):
        if self.pkg_states and rstate is not None:
            qs, new_rstate = self.brain.predict(self.session, x, outer_state=rstate, batch_size=batch_size)
            self.recent_rnn_state = new_rstate
            return qs
        return self.brain.predict(self.session, x, outer_state=None, batch_size=batch_size)

    def initialize(self, message, sender):
        my_id = super().initialize(message, sender)
        if isinstance(message, DQNRouterInitMsg):
            self._initModel(len(self.network), message.nn_type, message.getContents())
            self.batch_size = message.batch_size
            self.pkg_states = message.pkg_states
            if message.prioritized_xp:
                self.prioritized_xp = True
                self.memory = PrioritizedMemory(message.mem_capacity)
            else:
                self.memory = Memory(message.mem_capacity)
            self.initGraph(self.addr, self.network, self.neighbors, self.link_states)
            for n in self.neighbors.keys():
                self.outgoing_pkgs_num[n] = 0

        elif isinstance(message, RouterFinalizeInitMsg):
            self._announceLinkState()
        return my_id

    def _announceLinkState(self):
        announcement = self.mkLSAnnouncement(self.addr)
        self._broadcastAnnouncement(announcement, self.myAddress)
        self.seq_num += 1

    def _broadcastAnnouncement(self, announcement, sender):
        for n in self.neighbors.keys():
            if sender != self.network[n]:
                self.sendServiceMsg(self.network[n], announcement)

    def receiveServiceMsg(self, message, sender):
        super().receiveServiceMsg(message, sender)
        if isinstance(message, LinkStateAnnouncement):
            if self.processLSAnnouncement(message, self.network.keys()):
                self._broadcastAnnouncement(message, sender)
        # elif isinstance(message, NeighborsAdvice):
        #     n = self.network_inv[str(sender)]
        #     self.neighbors_advices[n] = (message.time, message.estimations)
        # elif isinstance(message, NeighborLoadStatus):
            # self.neighbors_load_statuses[self.network_inv[str(sender)]] = message.is_overloaded

    def breakLink(self, v):
        self.lsBreakLink(self.addr, v)
        self._announceLinkState()

    def restoreLink(self, v):
        self.lsRestoreLink(self.addr, v)
        self._announceLinkState()

    def isInitialized(self):
        return super().isInitialized() and (len(self.announcements) == len(self.network))

    def mkRewardMsg(self, pkg):
        d = pkg.dst
        best_estimate = 0
        if self.addr != d:
            s = self.brain.makeInputFromData(self.getState(pkg)[2])
            best_estimate = -np.amax(self._predict(s, pkg.rnn_state))
        return DQNRewardMsg(pkg.id, pkg.size, self.current_time, best_estimate, d)

    def mkSample(self, message, prev_state, sender):
        if isinstance(message, DQNRewardMsg):
            sent_time = prev_state[0]
            a = self.network_inv[str(sender)]
            s = self.brain.makeInputFromData(prev_state[2])
            rs = prev_state[1]
            r = -(message.estimate + (message.cur_time - sent_time))
            return (s, rs, a, r)
        else:
            raise Exception("Unsupported type of reward msg!")

    def _getAmatrix(self):
        gstate = np.ravel(nx.to_numpy_matrix(self.network_graph))
        for i, v in enumerate(gstate):
            gstate[i] = 0 if v == 0 else 1
        return gstate

    def getState(self, pkg):
        d = pkg.dst
        k = self.addr
        n = len(self.network)
        df = pd.DataFrame(columns=['dst', 'addr']+get_neighbors_cols(n)+get_amatrix_cols(n))
        basic_state_inp = np.zeros(n+2)
        basic_state_inp[0] = d
        basic_state_inp[1] = k
        amatrix = self._getAmatrix()
        basic_state_inp[2:] = amatrix[n*k:n*(k+1)]
        df.loc[0] = np.concatenate((basic_state_inp, amatrix))
        return (self.current_time, pkg.rnn_state, df)

    def _tellLoadStatus(self):
        lvl_avg = np.average(self.load_lvl_mavg, weights=LOAD_LVL_WEIGHTS)
        if self.queue_count <= 1 and lvl_avg < LOAD_LVL_LOW_THRESHOLD and self.ploho_flag:
            print("U {} VSE HOROSHO !!!".format(self.addr))
            self.ploho_flag = False
            for n in self.neighbors.keys():
                self.sendServiceMsg(self.network[n], NeighborLoadStatus(False))
        elif self.queue_count > 2 and lvl_avg > LOAD_LVL_HIGH_THRESHOLD and not self.ploho_flag:
            print("U {} VSE OCHEN HUEVO !!!".format(self.addr))
            self.ploho_flag = True
            for n in self.neighbors.keys():
                self.sendServiceMsg(self.network[n], NeighborLoadStatus(True))

    def act(self, state, pkg):
        s = self.brain.makeInputFromData(state[2])
        rs = state[1]
        pred = self._predict(s, rs)[0]
        if self.pkg_states:
            pkg.rnn_state = self.recent_rnn_state
        # dst = state[1][0]
        # self._adjustAdvices(dst, pred)
        res = -1
        while res not in self.neighbors.keys():
            res = soft_argmax(pred, self.temp)
        self.outgoing_pkgs_num[res] += 1
        self.load_lvl_mavg.appendleft(self.queue_count)
        # self._tellLoadStatus()
        # self._tellNeighborsIfFree()
        return res

    def observe(self, sample):
        s, rs, a, r = sample
        if self.outgoing_pkgs_num[a] > 0:
            self.outgoing_pkgs_num[a] -= 1
        if self.prioritized_xp:
            pred = self._predict(s, rs)[0][a]
            err = abs(r - pred)
            self.memory.add(err, sample)
        else:
            self.memory.add(sample)
        self.replay()
        self.steps += 1
        if self.temp > MIN_TEMP:
            self.temp = MAX_TEMP - (self.steps / DECAY_TEMP_STEPS) * (MAX_TEMP - MIN_TEMP)

    def replay(self):
        i_batch = self.memory.sample(self.batch_size)
        blen = len(i_batch)
        b_idxs = [b[0] for b in i_batch]
        batch = [b[1] for b in i_batch]

        states = stack_batch([l[0] for l in batch])
        rstates = stack_batch([l[1] for l in batch])
        actions = [l[2] for l in batch]
        values = [l[3] for l in batch]

        preds = self._predict(states, rstates, batch_size=blen)
        for i in range(blen):
            a = actions[i]
            error = abs(preds[i][a] - values[i])
            preds[i][a] = values[i]
            if self.prioritized_xp:
                self.memory.update(b_idxs[i], error)

        outer_states = None
        if self.pkg_states:
            outer_states = rstates
        self._train(states, preds, outer_state=outer_states, batch_size=blen,
                    save_old_state=True)
