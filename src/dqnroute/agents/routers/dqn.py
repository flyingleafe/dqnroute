import random
import math
import logging
import torch
import torch.nn as nn
import numpy as np
import networkx as nx
import pandas as pd
import pprint
import os

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
    def __init__(self, batch_size: int, mem_capacity: int, nodes: List[AgentId],
                 optimizer='rmsprop', brain=None, random_init=False, max_act_time=None,
                 additional_inputs=[], **kwargs):
        super().__init__(**kwargs)
        self.batch_size = batch_size
        self.memory = Memory(mem_capacity)
        self.additional_inputs = additional_inputs
        self.nodes = nodes
        self.max_act_time = max_act_time
        
        # added by Igor to add probability smoothing support:
        if "IGOR_TRAIN_PROBABILITY_SMOOTHING" in os.environ:
            self.probability_smoothing = float(os.environ["IGOR_TRAIN_PROBABILITY_SMOOTHING"])
        else:
            self.probability_smoothing = 0.0

        if brain is None:
            self.brain = self._makeBrain(additional_inputs=additional_inputs, **kwargs)
            if random_init:
                self.brain.init_xavier()
            else:
                # added by Igor to override the filename of the loaded model:
                if "IGOR_OVERRIDED_DQN_LOAD_FILENAME" in os.environ:
                    self.brain._label = os.environ["IGOR_OVERRIDED_DQN_LOAD_FILENAME"]
                
                self.brain.restore()
                self.log('Restored model {}'.format(self.brain._label))

        else:
            self.brain = brain

        self.optimizer = get_optimizer(optimizer)(self.brain.parameters())
        self.loss_func = nn.MSELoss()

    def route(self, sender: AgentId, pkg: Package, allowed_nbrs: List[AgentId]) -> Tuple[AgentId, List[Message]]:
        if self.max_act_time is not None and self.env.time() > self.max_act_time:
            return super().route(sender, pkg, allowed_nbrs)
        else:
            to, estimate, saved_state = self._act(pkg, allowed_nbrs)
            reward = self.registerResentPkg(pkg, estimate, to, saved_state)
            return to, [OutMessage(self.id, sender, reward)] if sender[0] != 'world' else []

    def handleMsgFrom(self, sender: AgentId, msg: Message) -> List[Message]:
        if isinstance(msg, RewardMsg):
            action, Q_new, prev_state = self.receiveReward(msg)
            self.memory.add((prev_state, action[1], -Q_new))
            self._replay()
            return []
        else:
            return super().handleMsgFrom(sender, msg)

    def _makeBrain(self, additional_inputs=[], **kwargs):
        return QNetwork(len(self.nodes), additional_inputs=additional_inputs, one_out=False, **kwargs)

    def _act(self, pkg: Package, allowed_nbrs: List[AgentId]):
        state = self._getNNState(pkg, allowed_nbrs)
        prediction = self._predict(state)[0]
        distr = softmax(prediction, MIN_TEMP)
        estimate = -np.dot(prediction, distr)

        to = -1
        while ('router', to) not in allowed_nbrs:
            to = sample_distr(distr)

        return ('router', to), estimate, state

    def _predict(self, x):
        self.brain.eval()
        return self.brain(*map(torch.from_numpy, x)).clone().detach().numpy()

    def _train(self, x, y):
        self.brain.train()
        self.optimizer.zero_grad()
        output = self.brain(*map(torch.from_numpy, x))
        loss = self.loss_func(output, torch.from_numpy(y))
        loss.backward()
        self.optimizer.step()
        return float(loss)

    def _getAddInput(self, tag, *args, **kwargs):
        if tag == 'amatrix':
            amatrix = nx.convert_matrix.to_numpy_array(
                self.network, nodelist=self.nodes, weight=self.edge_weight,
                dtype=np.float32)
            gstate = np.ravel(amatrix)
            return gstate
        else:
            raise Exception('Unknown additional input: ' + tag)

    def _getNNState(self, pkg: Package, nbrs: List[AgentId]):
        n = len(self.nodes)
        addr = np.array(self.id[1])
        dst = np.array(pkg.dst[1])

        neighbours = np.array(
            list(map(lambda v: v in nbrs, self.nodes)),
            dtype=np.float32)
        input = [addr, dst, neighbours]

        for inp in self.additional_inputs:
            tag = inp['tag']
            add_inp = self._getAddInput(tag)
            if tag == 'amatrix':
                add_inp[add_inp > 0] = 1
            input.append(add_inp)

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

    def _act(self, pkg: Package, allowed_nbrs: List[AgentId]):
        state = self._getNNState(pkg, allowed_nbrs)
        prediction = self._predict(state).flatten()
        distr = softmax(prediction, MIN_TEMP)
        
        # Igor: probability smoothing
        if len(distr) == 2:
            #print(distr)
            distr = (1 - self.probability_smoothing) * distr + self.probability_smoothing / 2
            #print(distr)
        
        to_idx = sample_distr(distr)
        estimate = -np.dot(prediction, distr)

        saved_state = [s[to_idx] for s in state]
        to = allowed_nbrs[to_idx]
        return to, estimate, saved_state

    def _nodeRepr(self, node):
        return np.array(node)

    def _getAddInput(self, tag, nbr):
        return super()._getAddInput(tag)

    def _getNNState(self, pkg: Package, nbrs: List[AgentId]):
        n = len(self.nodes)
        addr = self._nodeRepr(self.id[1])
        dst = self._nodeRepr(pkg.dst[1])

        get_add_inputs = lambda nbr: [self._getAddInput(inp['tag'], nbr)
                                      for inp in self.additional_inputs]

        input = [[addr, dst, self._nodeRepr(v[1])] + get_add_inputs(v)
                 for v in nbrs]
        return stack_batch(input)

    def _replay(self):
        states, _, values = self._sampleMemStacked()
        self._train(states, np.expand_dims(np.array(values, dtype=np.float32), axis=0))


class DQNRouterEmb(DQNRouterOO):
    """
    Variant of DQNRouter which uses graph embeddings instead of
    one-hot label encodings.
    """
    def __init__(self, embedding: Union[dict, Embedding], edges_num: int, **kwargs):
        # Those are used to only re-learn the embedding when the topology is changed
        self.prev_num_nodes = 0
        self.prev_num_edges = 0
        self.init_edges_num = edges_num
        self.network_initialized = False

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

    def networkStateChanged(self):
        num_nodes = len(self.network.nodes)
        num_edges = len(self.network.edges)

        if not self.network_initialized and num_nodes == len(self.nodes) and num_edges == self.init_edges_num:
            self.network_initialized = True

        if self.network_initialized and (num_edges != self.prev_num_edges or num_nodes != self.prev_num_nodes):
            self.prev_num_nodes = num_nodes
            self.prev_num_edges = num_edges
            self.embedding.fit(self.network, weight=self.edge_weight)
            # self.log(pprint.pformat(self.embedding._X), force=self.id[1] == 0)


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
    def _getAddInput(self, tag, nbr=None):
        if tag == 'work_status':
            return np.array(
                list(map(lambda n: self.network.nodes[n].get('works', False), self.nodes)),
                dtype=np.float32)
        if tag == 'working':
            nbr_works = 1 if self.network.nodes[nbr].get('works', False) else 0
            return np.array(nbr_works, dtype=np.float32)
        else:
            return super()._getAddInput(tag, nbr)


class DQNRouterConveyor(LSConveyorMixin, ConveyorRewardAgent, ConveyorAddInputMixin, DQNRouter):
    pass

class DQNRouterOOConveyor(LSConveyorMixin, ConveyorRewardAgent, ConveyorAddInputMixin, DQNRouterOO):
    pass

class DQNRouterEmbConveyor(LSConveyorMixin, ConveyorRewardAgent, ConveyorAddInputMixin, DQNRouterEmb):
    pass

