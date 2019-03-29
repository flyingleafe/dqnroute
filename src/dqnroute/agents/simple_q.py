from typing import List, Tuple, Dict
from .base import *
from ..messages import *
from ..utils import dict_min

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

    def addLink(self, to: int, params={}) -> List[Message]:
        msgs = super().addLink(to, params)
        for (u, dct) in self.Q.items():
            if to not in dct:
                dct[to] = 0 if u == to else 10
        return msgs

    def route(self, sender: int, pkg: Package) -> Tuple[int, List[Message]]:
        Qs = self._Q(pkg.dst)
        best_to, estimate = dict_min(Qs)
        now = self.currentTime()
        self.registerSentPkg(pkg, {'time_sent': self.currentTime(), 'dst': pkg.dst})

        if self.choice_policy == 'softmax':
            q = np.full(len(self.Q), -INFTY)
            for (node, val) in Qs.items():
                q[node] = -val
            to = -1
            while to not in self.out_neighbours:
                to = soft_argmax(q)
        else:
            to = best_to

        reward_msg = NetworkRewardMsg(pkg.id, now, estimate)
        return to, [OutMessage(sender, reward_msg)] if sender != -1 else []

    def handleServiceMsg(self, sender: int, msg: ServiceMessage) -> List[Message]:
        if isinstance(msg, RewardMsg):
            Q_new, saved_data = self.receiveReward(msg)
            dst = saved_data['dst']
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

class SimpleQRouterConveyor(SimpleQRouter, ConveyorRewardAgent):
    """
    Q-router which calculates rewards for conveyor routing setting
    """
    pass
