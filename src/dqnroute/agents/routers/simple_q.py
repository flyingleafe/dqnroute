import networkx as nx

from typing import List, Tuple, Dict
from ..base import *
from .link_state import *
from ...messages import *
from ...utils import dict_min

class SimpleQRouter(Router, RewardAgent):
    """
    A router which implements Q-routing algorithm
    """
    def __init__(self, env: DynamicEnv, learning_rate: float,
                 nodes: List[int], choice_policy: str = 'strict', **kwargs):
        super().__init__(env, **kwargs)
        self.learning_rate = learning_rate
        self.choice_policy = choice_policy
        self.Q = {u: {v: 0 if u == v else 10
                      for v in self.out_neighbours}
                  for u in nodes}

    def addLink(self, to: int, direction: str, params={}) -> List[Message]:
        msgs = super().addLink(to, direction, params)
        if direction != 'in':
            for (u, dct) in self.Q.items():
                if to not in dct:
                    dct[to] = 0 if u == to else 10
        return msgs

    def route(self, sender: int, pkg: Package) -> Tuple[int, List[Message]]:
        Qs = self._Q(pkg.dst)
        best_to, estimate = dict_min(Qs)
        reward_msg = self.registerResentPkg(pkg, estimate, pkg.dst)

        if self.choice_policy == 'softmax':
            q = np.full(len(self.Q), -INFTY)
            for (node, val) in Qs.items():
                q[node] = -val
            to = -1
            while to not in self.out_neighbours:
                to = soft_argmax(q)
        else:
            to = best_to

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
        res = {}
        for n in self.out_neighbours:
            res[n] = self.Q[d][n]
        return res

class SimpleQRouterNetwork(SimpleQRouter, NetworkRewardAgent):
    """
    Q-router which calculates rewards for computer routing setting
    """
    pass

class SimpleQRouterConveyor(SimpleQRouter, LinkStateRouter, ConveyorRewardAgent):
    """
    Q-router which calculates rewards for conveyor routing setting
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
        scheduled_stop_time = self.env.time() + self.env.stop_delay()
        msgs.append(OutConveyorMsg(StopTimeUpdMsg(scheduled_stop_time)))

        self.out_neighbours = old_neighbours
        return to, msgs

    def handleServiceMsg(self, sender: int, msg: ServiceMessage) -> List[Message]:
        if isinstance(msg, ConveyorServiceMsg):
            if isinstance(msg, ConveyorStartMsg):
                self.scheduled_stop_time = self.env.time()
            return []
        else:
            return super().handleServiceMsg(sender, msg)
