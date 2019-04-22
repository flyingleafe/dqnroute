import networkx as nx
import random

from typing import List, Tuple, Dict
from ..base import *
from .link_state import *
from ...messages import *
from ...utils import dict_min

class SimpleQRouter(Router, RewardAgent):
    """
    A router which implements Q-routing algorithm
    """
    def __init__(self, learning_rate: float, nodes: List[int], **kwargs):
        super().__init__(**kwargs)
        self.learning_rate = learning_rate
        self.nodes = nodes
        self.Q = {u: {v: 0 if u == v else random.randint(1, 10)
                      for v in self.out_neighbours}
                  for u in self.nodes}

    def addLink(self, to: int, direction: str, params={}) -> List[Message]:
        msgs = super().addLink(to, direction, params)
        if direction != 'in':
            for (u, dct) in self.Q.items():
                if to not in dct:
                    dct[to] = 0 if u == to else random.randint(1, 10)
        return msgs

    def route(self, sender: int, pkg: Package) -> Tuple[int, List[Message]]:
        Qs = self._Q(pkg.dst)
        to, estimate = dict_min(Qs)
        reward_msg = self.registerResentPkg(pkg, estimate, pkg.dst)

        return to, [OutMessage(self.id, sender, reward_msg)] if sender != -1 else []

    def handleServiceMsg(self, sender: int, msg: ServiceMessage) -> List[Message]:
        if isinstance(msg, RewardMsg):
            Q_new, dst = self.receiveReward(msg)
            self.Q[dst][sender] += self.learning_rate * (Q_new - self.Q[dst][sender])
            return []
        else:
            return super().handleServiceMsg(sender, msg)

    def _Q(self, d: int) -> Dict[int, float]:
        """
        Returns a dict which only includes available neighbours
        """
        return {n: self.Q[d][n] for n in self.out_neighbours}


class PredictiveQRouter(SimpleQRouter, RewardAgent):
    def __init__(self, beta: float, gamma: float, **kwargs):
        super().__init__(**kwargs)
        self.beta = beta
        self.gamma = gamma
        self.B = deepcopy(self.Q)
        self.R = {u: {v: 0 for v in self.out_neighbours}
                  for u in self.nodes}
        self.U = {u: {v: 0 for v in self.out_neighbours}
                  for u in self.nodes}

    def addLink(self, to: int, direction: str, params={}) -> List[Message]:
        msgs = super().addLink(to, direction, params)
        if direction != 'in':
            for u in self.nodes:
                if to not in self.B[u]:
                    self.B[u][to] = 0 if u == to else self.Q[u][to]
                if to not in self.R[u]:
                    self.R[u][to] = 0
                if to not in self.U[u]:
                    self.U[u][to] = self.env.time()
        return msgs

    def route(self, sender: int, pkg: Package) -> Tuple[int, List[Message]]:
        Qs = self._Q(pkg.dst)
        Qs_altered = self._Q_altered(pkg.dst)
        to, _ = dict_min(Qs_altered)
        estimate = min(Qs.values())
        reward_msg = self.registerResentPkg(pkg, estimate, pkg.dst)

        return to, [OutMessage(self.id, sender, reward_msg)] if sender != -1 else []

    def handleServiceMsg(self, sender: int, msg: ServiceMessage) -> List[Message]:
        if isinstance(msg, RewardMsg):
            Q_new, dst = self.receiveReward(msg)
            dQ = Q_new - self.Q[dst][sender]
            self.Q[dst][sender] += self.learning_rate * dQ
            self.B[dst][sender] = min(self.B[dst][sender], self.Q[dst][sender])

            now = self.env.time()
            if dQ < 0:
                dR = dQ / (now - self.U[dst][sender])
                self.R[dst][sender] += self.beta * dR
            elif dQ > 0:
                self.R[dst][sender] *= self.gamma

            self.U[dst][sender] = now
            return []
        else:
            return super().handleServiceMsg(sender, msg)

    def _Q_altered(self, d: int) -> Dict[int, float]:
        """
        Returns estimates for all available neighbours
        """
        now = self.env.time()
        res = {}
        for n in self.out_neighbours:
            dt = now - self.U[d][n]
            res[n] = max(self.Q[d][n] + dt * self.R[d][n], self.B[d][n])
        return res

class SimpleQRouterNetwork(NetworkRewardAgent, SimpleQRouter):
    """
    Q-router which calculates rewards for computer routing setting
    """
    pass

class SimpleQRouterConveyor(LSConveyorMixin, ConveyorRewardAgent, SimpleQRouter, LinkStateRouter):
    """
    Q-router which calculates rewards for conveyor routing setting
    """
    pass

class PredictiveQRouterNetwork(NetworkRewardAgent, PredictiveQRouter):
    pass

class PredictiveQRouterConveyor(LSConveyorMixin, ConveyorRewardAgent, PredictiveQRouter, LinkStateRouter):
    pass
