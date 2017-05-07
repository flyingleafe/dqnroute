import random
import math
import numpy as np

from keras.models import Sequential, load_model
from keras.layers import *
from keras.optimizers import *

from messages import *
from router import *

BATCH_SIZE = 1

MAX_EPSILON = 1
MIN_EPSILON = 0.01
LAMBDA = 0.01

class DQNRouter(QRouter):
    def __init__(self):
        super().__init__()
        self.brain = None

    # def _createModel(self, input_dim, output_dim):
    #     model = Sequential()
    #     model.add(Dense(output_dim=64, activation='relu', input_dim=input_dim))
    #     model.add(Dense(output_dim=output_dim, activation='linear'))

    #     opt = RMSprop()
    #     model.compile(loss='mse', optimizer=opt)

    #     return model

    def _train(self, x, y, epoch=1, verbose=0):
        self.brain.fit(x, y, batch_size=BATCH_SIZE, nb_epoch=epoch, verbose=verbose)

    def initialize(self, message, sender):
        super().initialize(message, sender)
        if isinstance(message, DQNRouterInitMsg):
            n = len(self.neighbors)
            self.brain = load_model(message.model_file)

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
            a = list(self.neighbors.keys()).index(sender_addr)
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
        return (self.current_time, res.reshape((1, self.state_dim)))

    def act(self, state):
        if random.random() < self.epsilon:
            return random.choice(list(self.neighbors.keys()))
        else:
            s = state[1]
            prediction = np.argmin(self.brain.predict(s))
            return list(self.neighbors.keys())[prediction]

    def observe(self, sample):
        (s, a, r) = sample
        p = self.brain.predict(s).flatten()
        p[a] = r
        x = np.zeros((1, self.state_dim))
        y = np.zeros((1, len(self.neighbors)))
        x[0] = s
        y[0] = p
        self._train(x, y)
        self.steps += 1
        self.epsilon = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * math.exp(-LAMBDA * self.steps)
