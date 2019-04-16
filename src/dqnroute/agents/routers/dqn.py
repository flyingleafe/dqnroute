import random
import math
import logging
import torch
import torch.nn as nn
import numpy as np
import networkx as nx
import pandas as pd

from typing import List, Tuple, Dict
from ..base import *
from .link_state import *
from ...constants import DQNROUTE_LOGGER
from ...messages import *
from ...memory import *
from ...utils import *
from ...networks import *

MIN_TEMP = 1.5

logger = logging.getLogger(DQNROUTE_LOGGER)

class DQNRouter(LinkStateRouter, RewardAgent):
    """
    A router which implements DQN-routing algorithm
    """
    def __init__(self, batch_size: int, mem_capacity: int, nodes: List[int],
                 embedding=None, optimizer='rmsprop', additional_inputs=[], **kwargs):
        super().__init__(**kwargs)
        self.batch_size = batch_size
        self.memory = Memory(mem_capacity)
        self.additional_inputs = additional_inputs
        self.nodes = nodes

        if embedding is None:
            self.embedding = None
            embedding_dim = None
        else:
            self.embedding = HOPEEmbedding(**embedding)
            embedding_dim = self.embedding.dim

        self.brain = QNetwork(len(self.nodes), additional_inputs=additional_inputs,
                              embedding_dim=embedding_dim, **kwargs)

        self.optimizer = get_optimizer(optimizer)(self.brain.parameters())
        self.loss_func = nn.MSELoss()
        self.brain.restore()
        logger.info('Restored model ' + self.brain._label)

    def route(self, sender: int, pkg: Package) -> Tuple[int, List[Message]]:
        state = self._getNNState(pkg)
        prediction = self._predict(state).flatten()
        to_idx = soft_argmax(prediction, MIN_TEMP)
        estimate = -np.max(prediction)

        saved_state = [s[to_idx] for s in state]
        to = sorted(list(self.out_neighbours))[to_idx]

        reward = self.registerResentPkg(pkg, estimate, saved_state)
        return to, [OutMessage(self.id, sender, reward)] if sender != -1 else []

    def handleServiceMsg(self, sender: int, msg: ServiceMessage) -> List[Message]:
        if isinstance(msg, RewardMsg):
            Q_new, prev_state = self.receiveReward(msg)
            self.memory.add((prev_state, sender, -Q_new))
            self._replay()
            return []
        else:
            return super().handleServiceMsg(sender, msg)

    def networkComplete(self):
        return len(self.network.nodes) == len(self.nodes)

    def networkInit(self):
        if self.embedding is not None:
            self.embedding.fit(self.network, weight=self.edge_weight)

    def _predict(self, x):
        self.brain.eval()
        return self.brain(*map(torch.from_numpy, x))\
                   .clone().detach().numpy()

    def _train(self, x, y):
        self.brain.train()
        self.optimizer.zero_grad()
        output = self.brain(*map(torch.from_numpy, x))
        loss = self.loss_func(output, torch.from_numpy(y))
        loss.backward()
        self.optimizer.step()
        return float(loss)

    def _getAddInput(self, tag):
        if tag == 'amatrix':
            amatrix = nx.convert_matrix.to_numpy_array(
                self.network, nodelist=self.nodes,
                dtype=np.float32)
            gstate = np.ravel(amatrix)
            gstate[gstate > 0] = 1
            return gstate
        else:
            raise Exception('Unknown additional input: ' + tag)

    def _nodeRepr(self, node):
        if self.embedding is None:
            return np.array(node)
        else:
            return self.embedding.get_embedding(node).astype(np.float32)

    def _getNNState(self, pkg: Package):
        n = len(self.nodes)

        addr = self._nodeRepr(self.id)
        dst = self._nodeRepr(pkg.dst)

        neighbours = np.array(
            list(map(lambda v: v in self.out_neighbours, self.nodes)),
            dtype=np.float32)

        add_inputs = [self._getAddInput(inp['tag']) for inp in self.additional_inputs]

        input = [[addr, dst, self._nodeRepr(v)] + add_inputs
                 for v in sorted(list(self.out_neighbours))]
        return stack_batch(input)

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

        self._train(states, np.array(values, dtype=np.float32))

class DQNRouterNetwork(NetworkRewardAgent, DQNRouter):
    """
    DQN-router which calculates rewards for computer routing setting
    """
    pass

class DQNRouterConveyor(LSConveyorMixin, ConveyorRewardAgent, DQNRouter):
    """
    DQN-router which calculates rewards for conveyors setting
    """
    def _getAddInput(self, tag):
        if tag == 'work_status':
            return np.array(
                list(map(lambda n: self.network.nodes[n].get('works', False), self.nodes)),
                dtype=np.float32)
        else:
            return super()._getAddInput(tag)
