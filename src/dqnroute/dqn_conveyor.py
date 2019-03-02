import random
import math
import numpy as np
import networkx as nx
import tensorflow as tf
import pandas as pd

from collections import deque

from .messages import *
from .conveyor import *
from .memory import *
from .utils import *
from .networks import get_qnetwork_class

MAX_TEMP = 10.0
MIN_TEMP = 1.5
DECAY_TEMP_STEPS = 60000

class DQNConveyor(QConveyor):
    def __init__(self):
        super().__init__()
        self.batch_size = 1
        self.pkg_states = False
        self.brain = None
        self.memory = None
        self.prioritized_xp = False
        self.temp = MIN_TEMP
        self.steps = 0
        self.err_mavg = None
        self.session = None
        self.load_lvl_mavg = deque([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 10)
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
        if isinstance(message, DQNConveyorInitMsg):
            self._initModel(len(self.all_sections), message.nn_type, message.getContents())
            self.batch_size = message.batch_size
            self.pkg_states = message.pkg_states
            if message.prioritized_xp:
                self.prioritized_xp = True
                self.memory = PrioritizedMemory(message.mem_capacity)
            else:
                self.memory = Memory(message.mem_capacity)

        return my_id

    def mkRewardMsg(self, bag_event):
        bag = bag_event.getContents()
        d = bag.dst
        section = bag_event.section
        reward = self._computeReward(bag_event)
        best_estimate = 0
        if self.addr != d:
            s = self.brain.makeInputFromData(self.getState(bag, section)[2])
            best_estimate = np.amax(self._predict(s))
        return DQNConveyorRewardMsg(bag.id, section, reward, best_estimate)

    def mkSample(self, message, prev_state, sender):
        if isinstance(message, DQNConveyorRewardMsg):
            a = message.section
            s = self.brain.makeInputFromData(prev_state[2])
            rs = prev_state[1]
            r = message.reward
            q = message.estimate
            return (s, rs, a, r, q)
        else:
            raise Exception("Unsupported type of reward msg!")

    def _getAmatrix(self):
        gstate = np.ravel(nx.to_numpy_matrix(self.network_graph))
        for i, v in enumerate(gstate):
            gstate[i] = 0 if v == 0 else 1
        return gstate

    def getState(self, bag, section):
        d = bag.dst
        k = section
        n = len(self.all_sections)
        df = pd.DataFrame(columns=['dst', 'addr']+get_neighbors_cols(n)+
                          get_work_status_cols(n)+get_amatrix_cols(n))
        basic_state_inp = np.zeros(2*n+2)
        basic_state_inp[0] = d
        basic_state_inp[1] = k
        off = 2
        for nb in self._possibleNeighbors(section, d):
            basic_state_inp[off+nb] = 1
        off += n
        basic_state_inp[off:off+n] = self._neighborsWorkingEnc()
        amatrix = self._getAmatrix()
        df.loc[0] = np.concatenate((basic_state_inp, amatrix))
        return (self.current_time, bag.rnn_state, df)

    def act(self, state, bag):
        st = state[2]
        rs = state[1]
        section = st['addr'][0]
        dst = st['dst'][0]
        s = self.brain.makeInputFromData(st)
        pred = self._predict(s, rs)[0]
        if self.pkg_states:
            pkg.rnn_state = self.recent_rnn_state
        res = -1
        while res not in self._possibleNeighbors(section, dst):
            res = soft_argmax(pred, self.temp)
        # self.load_lvl_mavg.appendleft(self.queue_count)
        return res

    def observe(self, sample):
        s, rs, a, r, q = sample
        if self.prioritized_xp:
            Q_val = r + q
            pred = self._predict(s, rs)[0][a]
            err = abs(Q_val - pred)
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
        values = [l[3] + l[4] for l in batch]

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
