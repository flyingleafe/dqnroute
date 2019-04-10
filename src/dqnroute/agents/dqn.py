import random
import math
import logging
import torch
import torch.nn as nn
import numpy as np
import networkx as nx
import pandas as pd

from typing import List, Tuple, Dict
from .base import *
from .link_state import LinkStateRouter
from ..constants import DQNROUTE_LOGGER
from ..messages import *
from ..memory import *
from ..utils import *
from ..networks import QNetwork, get_optimizer

MIN_TEMP = 1.5

logger = logging.getLogger(DQNROUTE_LOGGER)

class DQNRouter(LinkStateRouter, RewardAgent):
    """
    A router which implements DQN-routing algorithm
    """
    def __init__(self, env: DynamicEnv, batch_size: int, mem_capacity: int,
                 optimizer='rmsprop', **kwargs):
        super().__init__(env, **kwargs)
        self.batch_size = batch_size
        self.memory = Memory(mem_capacity)

        self.brain = QNetwork(len(self.network.nodes), **kwargs)
        self.optimizer = get_optimizer(optimizer)(self.brain.parameters())
        self.loss_func = nn.MSELoss()
        self.brain.restore()
        logger.info('Restored model ' + self.brain._label)

    def route(self, sender: int, pkg: Package) -> Tuple[int, List[Message]]:
        state = self._getNNState(pkg)
        prediction = self._predict(state)[0]

        to = -1
        while to not in self.out_neighbours:
            to = soft_argmax(prediction, MIN_TEMP)

        estimate = -np.max(prediction)
        reward = self.registerResentPkg(pkg, estimate, state)
        return to, [OutMessage(self.id, sender, reward)] if sender != -1 else []

    def handleServiceMsg(self, sender: int, msg: ServiceMessage) -> List[Message]:
        if isinstance(msg, RewardMsg):
            Q_new, prev_state = self.receiveReward(msg)
            self.memory.add((prev_state, sender, -Q_new))
            self._replay()
            return []
        else:
            return super().handleServiceMsg(sender, msg)

    def _predict(self, x):
        return self.brain(*map(torch.from_numpy, x))\
                   .clone().detach().numpy()

    def _train(self, x, y):
        self.optimizer.zero_grad()
        output = self.brain(*map(torch.from_numpy, x))
        loss = self.loss_func(output, torch.from_numpy(y))
        loss.backward()
        self.optimizer.step()
        return float(loss)

    def _getAmatrix(self):
        amatrix = nx.convert_matrix.to_numpy_array(
            self.network, nodelist=sorted(self.network.nodes),
            dtype=np.float32)
        gstate = np.ravel(amatrix)
        gstate[gstate > 0] = 1
        return gstate

    def _getNNState(self, pkg: Package):
        n = len(self.network.nodes)
        addr = np.array(self.id)
        dst = np.array(pkg.dst)
        neighbours = np.array(list(map(lambda v: v in self.out_neighbours,
                                       sorted(self.network.nodes))),
                              dtype=np.float32)
        amatrix = self._getAmatrix()
        return (addr, dst, neighbours, amatrix)

    def _replay(self):
        """
        Fetches a batch of samples from the memory and fits against them
        """
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
            preds[i][a] = values[i]

        self._train(states, preds)

class DQNRouterNetwork(DQNRouter, NetworkRewardAgent):
    """
    DQN-router which calculates rewards for computer routing setting
    """
    pass

class DQNRouterConveyor(DQNRouter, ConveyorRewardAgent):
    """
    DQN-router which calculates rewards for computer routing setting
    """
    def route(self, sender: int, pkg: Package) -> Tuple[int, List[Message]]:
        """
        Makes sure that bags are not sent to the path which can not lead to
        the destination
        """
        old_neighbours = self.out_neighbours
        filter_func = lambda v: nx.has_path(self.network, v, pkg.dst)
        self.out_neighbours = set(filter(filter_func, old_neighbours))

        to, msgs = super().route(sender, pkg)
        self.out_neighbours = old_neighbours
        return to, msgs

