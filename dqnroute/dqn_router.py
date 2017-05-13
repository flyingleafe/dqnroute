import random
import math
import numpy as np
import networkx as nx
import tensorflow as tf

from collections import deque

from messages import *
from router import *
from memory import *
from q_network import Qnetwork
from router_mixins import LinkStateHolder

MAX_EPSILON = 1
MIN_EPSILON = 0.01
LAMBDA = 0.01

MAX_TEMP = 10.0
MIN_TEMP = 1.0
DECAY_TEMP_STEPS = 60000

class DQNRouter(QRouter, LinkStateHolder):
    def __init__(self):
        super().__init__()
        LinkStateHolder.__init__(self)
        self.batch_size = 1
        self.brain = None
        self.memory = None
        self.prioritized_xp = False
        self.temp = MIN_TEMP
        self.steps = 0
        self.err_mavg = None
        self.session = None
        self.outgoing_pkgs_num = {}

    def _initModel(self, n, path):
        tf.reset_default_graph()
        rnn_cell = tf.contrib.rnn.BasicLSTMCell(num_units=64)
        self.brain = Qnetwork(n, rnn_cell, 'mda')
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        self.session = tf.Session()
        # load model
        self.session.run(init)
        if path is not None:
            self.temp = MIN_TEMP
            ckpt = tf.train.get_checkpoint_state(path)
            print(ckpt)
            saver.restore(self.session, ckpt.model_checkpoint_path)
        else:
            self.temp = MAX_TEMP
            print('No pre-trained model loaded')

    def _mkFeedDict(self, x, y=None):
        Nb, Ab, Db, Mb = x
        feed_dict={self.brain.neighbors_input:Nb, self.brain.addr_input:Ab,
                   self.brain.dst_input:Db, self.brain.amatrix_input:Mb}
        if y is not None:
            feed_dict[self.brain.target] = y
        return feed_dict

    def _train(self, x, y):
        self.session.run(self.brain.updateModel, self._mkFeedDict(x, y))

    def _predict(self, x):
        return self.session.run(self.brain.Qout, self._mkFeedDict(x))

    def initialize(self, message, sender):
        my_id = super().initialize(message, sender)
        if isinstance(message, DQNRouterInitMsg):
            self._initModel(len(self.network), message.model_file)
            self.batch_size = message.batch_size
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
            best_estimate = -np.amax(self._predict(reverse_input(s)))
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
        res = -1
        while res not in self.neighbors.keys():
            pred = self._predict(s)[0]
            # print(s)
            # print(pred)
            res = soft_argmax(pred, self.temp)
        self.outgoing_pkgs_num[res] += 1
        return res

    def observe(self, sample):
        s, a, r = sample
        if self.outgoing_pkgs_num[a] > 0:
            self.outgoing_pkgs_num[a] -= 1
        if self.prioritized_xp:
            pred = self._predict(reverse_input(s))[0][a]
            err = abs(r - a)
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
        actions = [l[1] for l in batch]
        values = [l[2] for l in batch]

        preds = self._predict(states)
        for i in range(blen):
            a = actions[i]
            error = abs(preds[i][a] - values[i])
            preds[i][a] = values[i]
            if self.prioritized_xp:
                self.memory.update(b_idxs[i], error)

        self._train(states, preds)

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

def stack_batch(batch):
    n = len(batch[0])
    ss = [None]*n
    for i in range(n):
        ss[i] = np.vstack([b[i] for b in batch])
    return ss

def uni(n):
    return np.full(n, 1.0/n)

def softmax(x, t=1.0):
    ax = np.array(x) / t
    ax -= np.amax(ax)
    e = np.exp(ax)
    sum = np.sum(e)
    if sum == 0:
        return uni(len(ax))
    return e / np.sum(e, axis=0)

def soft_argmax(arr, t=1.0):
    return np.random.choice(np.arange(len(arr)), p=softmax(arr, t=t))
