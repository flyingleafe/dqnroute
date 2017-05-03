import numpy as np

from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *

from messages import *
from router import *

def DQNRouter(QRouter):
    def __init__(self):
        super().__init__()
        self.brain = None
        self.q_link_states = None
        self.ZEROS_ARR = None

    def _createModel(self, input_dim, output_dim):
        model = Sequential()
        model.add(Dense(output_dim=64, activation='relu', input_dim=input_dim))
        model.add(Dense(output_dim=output_dim, activation='linear'))

        opt = RMSProp()
        model.compile(loss='mse', optimizer=opt)

        return model

    def initialize(self, message, sender):
        super().initialize(message, sender)
        if isinstance(message, DQNRouterInitMsg):
            n = len(self.neighbors)
            state_dim = 2*n + len(self.network)
            self.brain = self._createModel(state_dim, n)
            self.q_link_states = np.zeros(2*n)
            self.ZEROS_ARR = np.zeros(len(self.network) + 1)

    def mkRewardMsg(self, pkg):
        d = pkg.dst
        best_estimate = 0
        if self.addr != d:
            s = self.getState(pkg)[1]
            best_estimate = np.amax(self.brain.predict(s))
        return DQNRewardMsg(pkg.id, pkg.size, self.current_time, best_estimate, d)

    def mkSample(self, message, prev_state, sender):
        if isinstance(message, DQNRewardMsg):
            sent_time = prev_state[0]
            sender_addr = self.network_inv[str(sender)]
            a = self.neighbors.keys().index(sender_addr)
            s = prev_state[1]
            r = message.estimate + (message.cur_time - sent_time)
            return (s, a, r)
        else:
            raise Exception("Unsupported type of reward msg!")

    def getState(self, pkg):
        d = pkg.dst
        size = pkg.size
        res = np.concatenate((self.ZEROS_ARR, self.q_link_states))
        res[d] = 1
        res[len(self.network)] = size
        return (self.current_time, res)

    def act(self, state):
        s = state[1]
        prediction = np.argmax(self.brain.predict(s))
        return self.neighbors.keys()[prediction]

    def observe(self, sample):
         pass
