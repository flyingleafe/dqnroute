import random
import math
import logging
import torch
import torch.nn as nn
import numpy as np
import networkx as nx
import pandas as pd

from typing import List, Tuple, Dict, Union
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
                 optimizer='rmsprop', brain=None, additional_inputs=[], **kwargs):
        super().__init__(**kwargs)
        self.batch_size = batch_size
        self.memory = Memory(mem_capacity)
        self.additional_inputs = additional_inputs
        self.nodes = nodes

        if brain is None:
            self.brain = self._makeBrain(additional_inputs=additional_inputs, **kwargs)
            self.brain.restore()
            logger.info('Restored model ' + self.brain._label)
        else:
            self.brain = brain

        self.optimizer = get_optimizer(optimizer)(self.brain.parameters())
        self.loss_func = nn.MSELoss()

    def route(self, sender: int, pkg: Package) -> Tuple[int, List[Message]]:
        to, estimate, saved_state = self._act(pkg)
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

    def _makeBrain(self, additional_inputs=[], **kwargs):
        return QNetwork(len(self.nodes), additional_inputs=additional_inputs,
                        one_out=False, **kwargs)

    def _act(self, pkg: Package):
        state = self._getNNState(pkg)
        prediction = self._predict(state)[0]

        to = -1
        while to not in self.out_neighbours:
            to = soft_argmax(prediction, MIN_TEMP)

        estimate = -np.max(prediction)
        return to, estimate, state

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

    def _getNNState(self, pkg: Package, nbrs=None):
        n = len(self.nodes)

        if nbrs is None:
            nbrs = self.out_neighbours

        addr = np.array(self.id)
        dst = np.array(pkg.dst)

        neighbours = np.array(
            list(map(lambda v: v in nbrs, self.nodes)),
            dtype=np.float32)
        input = [addr, dst, neighbours]

        for inp in self.additional_inputs:
            input.append(self._getAddInput(inp['tag']))

        return tuple(input)

    def _sampleMemStacked(self):
        """
        Samples a batch of episodes from memory and stacks
        states, actions and values from a batch together
        """
        i_batch = self.memory.sample(self.batch_size)
        batch = [b[1] for b in i_batch]

        states = stack_batch([l[0] for l in batch])
        actions = [l[1] for l in batch]
        values = [l[2] for l in batch]

        return states, actions, values

    def _replay(self):
        """
        Fetches a batch of samples from the memory and fits against them
        """
        states, actions, values = self._sampleMemStacked()
        preds = self._predict(states)

        for i in range(self.batch_size):
            a = actions[i]
            preds[i][a] = values[i]

        self._train(states, preds)


class DQNRouterOO(DQNRouter):
    """
    Variant of DQN router which uses Q-network with scalar output
    """
    def _makeBrain(self, additional_inputs=[], **kwargs):
        return QNetwork(len(self.nodes), additional_inputs=additional_inputs,
                        one_out=True, **kwargs)

    def _act(self, pkg: Package):
        state = self._getNNState(pkg)
        prediction = self._predict(state).flatten()
        to_idx = soft_argmax(prediction, MIN_TEMP)
        estimate = -np.max(prediction)

        saved_state = [s[to_idx] for s in state]
        to = sorted(list(self.out_neighbours))[to_idx]
        return to, estimate, saved_state

    def _nodeRepr(self, node):
        return np.array(node)

    def _getNNState(self, pkg: Package, nbrs=None):
        n = len(self.nodes)

        if nbrs is None:
            nbrs = sorted(list(self.out_neighbours))

        addr = self._nodeRepr(self.id)
        dst = self._nodeRepr(pkg.dst)

        add_inputs = [self._getAddInput(inp['tag']) for inp in self.additional_inputs]

        input = [[addr, dst, self._nodeRepr(v)] + add_inputs
                 for v in sorted(list(self.out_neighbours))]
        return stack_batch(input)

    def _replay(self):
        states, _, values = self._sampleMemStacked()
        self._train(states, np.array(values, dtype=np.float32))


class DQNRouterEmb(DQNRouterOO):
    """
    Variant of DQNRouter which uses graph embeddings instead of
    one-hot label encodings.
    """
    def __init__(self, embedding: Union[dict, Embedding], **kwargs):
        # Those are used to only re-learn the embedding when the topology is changed
        self.prev_num_nodes = 0
        self.prev_num_edges = 0

        if type(embedding) == dict:
            self.embedding = get_embedding(**embedding)
        else:
            self.embedding = embedding

        super().__init__(**kwargs)

    def _makeBrain(self, additional_inputs=[], **kwargs):
        return QNetwork(len(self.nodes), additional_inputs=additional_inputs,
                        embedding_dim=self.embedding.dim, one_out=True, **kwargs)

    def _nodeRepr(self, node):
        return self.embedding.transform(node).astype(np.float32)

    def networkComplete(self):
        return len(self.network.nodes) == len(self.nodes)

    def networkInit(self):
        if self.embedding is not None:
            num_nodes = len(self.network.nodes)
            num_edges = len(self.network.edges)
            if num_edges != self.prev_num_edges or num_nodes != self.prev_num_nodes:
                self.prev_num_nodes = num_nodes
                self.prev_num_edges = num_edges
                self.embedding.fit(self.network, weight=self.edge_weight)


class DQNRouterNetwork(NetworkRewardAgent, DQNRouter):
    pass

class DQNRouterOONetwork(NetworkRewardAgent, DQNRouterOO):
    pass

class DQNRouterEmbNetwork(NetworkRewardAgent, DQNRouterEmb):
    pass


class ConveyorAddInputMixin:
    """
    Mixing which adds conveyor-specific additional NN inputs support
    """
    def _getAddInput(self, tag):
        if tag == 'work_status':
            return np.array(
                list(map(lambda n: self.network.nodes[n].get('works', False), self.nodes)),
                dtype=np.float32)
        else:
            return super()._getAddInput(tag)


class DQNRouterConveyor(LSConveyorMixin, ConveyorRewardAgent, ConveyorAddInputMixin, DQNRouter):
    pass

class DQNRouterOOConveyor(LSConveyorMixin, ConveyorRewardAgent, ConveyorAddInputMixin, DQNRouterOO):
    pass

class DQNRouterEmbConveyor(LSConveyorMixin, ConveyorRewardAgent, ConveyorAddInputMixin, DQNRouterEmb):
    pass
