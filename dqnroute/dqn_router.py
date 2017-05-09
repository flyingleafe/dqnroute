import random
import math
import numpy as np
import networkx as nx

from keras.models import Sequential, load_model
from keras.layers import *
from keras.optimizers import *

from messages import *
from router import *
from router_mixins import LinkStateHolder

BATCH_SIZE = 1

MAX_EPSILON = 1
MIN_EPSILON = 0.01
LAMBDA = 0.01

class DQNRouter(QRouter, LinkStateHolder):
    def __init__(self):
        super().__init__()
        LinkStateHolder.__init__(self)
        self.brain = None

    # def _createModel(self, input_dim, output_dim):
    #     model = Sequential()
    #     model.add(Dense(output_dim=64, activation='relu', input_dim=input_dim))
    #     model.add(Dense(output_dim=output_dim, activation='linear'))

    #     opt = RMSprop()
    #     model.compile(loss='mse', optimizer=opt)

    #     return model

    def _train(self, x, y, epoch=1, verbose=0):
        self.brain.fit(x, y, batch_size=BATCH_SIZE, epochs=epoch, verbose=verbose)

    def initialize(self, message, sender):
        my_id = super().initialize(message, sender)
        if isinstance(message, DQNRouterInitMsg):
            model_file = message.model_file.format(self.addr)
            self.brain = load_model(model_file)
            self.initGraph(self.addr, self.network, self.neighbors, self.link_states)
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
            s = self._getInputs(self.getState(pkg)[1])
            best_estimate = -np.amax(self.brain.predict(reverse_input(s)))
        return DQNRewardMsg(pkg.id, pkg.size, self.current_time, best_estimate, d)

    def mkSample(self, message, prev_state, sender):
        if isinstance(message, DQNRewardMsg):
            sent_time = prev_state[0]
            a = self.network_inv[str(sender)]
            s = self._getInputs(prev_state[1])
            r = -(message.estimate + (message.cur_time - sent_time))
            return (s, a, r)
        else:
            raise Exception("Unsupported type of reward msg!")

    def getState(self, pkg):
        d = pkg.dst
        k = self.addr
        gstate = np.ravel(nx.to_numpy_matrix(self.network_graph))
        for i, v in enumerate(gstate):
            gstate[i] = 0 if v == 0 else 1
        return (self.current_time, (d, k, gstate))

    def _getInputs(self, state):
        n = len(self.network)
        d, k, gstate = state
        addr_arr = np.zeros(n)
        addr_arr[k] = 1
        dst_arr = np.zeros(n)
        dst_arr[d] = 1
        neighbors_arr = gstate[k*n : (k+1)*n]
        return [neighbors_arr, addr_arr, dst_arr, gstate]

    def act(self, state):
        _s = self._getInputs(state[1])
        s = reverse_input(_s)
        return np.argmax(self.brain.predict(s))

    def observe(self, sample):
        (s, a, r) = sample
        x = reverse_input(s)
        y = self.brain.predict(x)
        y[0][a] = r
        self._train(x, y)

def transpose_arr(arr):
    sh = arr.shape
    if len(sh) == 1:
        sh = (sh[0], 1)
    a, b = sh
    if a is None:
        a = 1
    if b is None:
        b = 1
    return arr.reshape((b, a))

def reverse_input(inp):
    return list(map(transpose_arr, inp))
