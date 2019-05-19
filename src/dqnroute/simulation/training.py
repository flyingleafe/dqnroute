import networkx as nx
import os
import random

from typing import List
from simpy import Environment, Event, Interrupt
from ..event_series import *
from ..messages import *
from ..agents import *
from ..utils import *
from .common import *


class TrainerRewardAgent(RewardAgent):
    """
    Computes rewards based only on path costs
    """
    def _computeReward(self, msg: TrainingRewardMsg, old_reward_data):
        from_node = msg.origin
        return self.pathCost(from_node)

    def _mkReward(self, pkg: Package, Q_estimate: float, reward_data) -> TrainingRewardMsg:
        orig = super()._mkReward(pkg, Q_estimate, reward_data)
        return TrainingRewardMsg(orig)


def TrainingRouterClass(Trainee: DQNRouter, Trainer: Router,
                        strict_guidance=True, strict_rewards=True, **kwargs):
    def __init__(self, brain, loss_series, **kwargs):
        super(Trainee, self).__init__(brain=brain, **kwargs)
        self.loss_series = loss_series

    def _train(self, x, y):
        loss = super(Trainee, self)._train(x, y)
        self.loss_series.logEvent(self.env.time(), loss)
        return loss

    def route(self, sender: AgentId, pkg: Package, allowed_nbrs: List[AgentId]) -> Tuple[AgentId, List[Message]]:
        to, trainer_msgs = Trainer.route(self, sender, pkg, allowed_nbrs)
        allowed_trainee_nbrs = [to] if strict_guidance else allowed_nbrs
        to, trainee_msgs = Trainee.route(self, sender, pkg, allowed_trainee_nbrs)

        trainer_estimate = self.pathCost(pkg.dst, through=to)
        for msg in trainee_msgs:
            if isinstance(msg, RewardMsg):
                msg.Q_estimate = trainer_estimate

        return to, trainer_msgs + trainee_msgs

    classname = 'Training_{}_{}'.format(Trainee.__name__, Trainer.__name__)
    bases = (TrainerRewardAgent, Trainee, Trainer) if strict_rewards else (Trainee, Trainer)
    methods = {'route': route, '__init__': __init__, '_train': _train}

    return type(classname, bases, methods)

def run_training(RunnerClass, router_type, training_router_type,
                 pkg_num=None, breaks_num=0, max_breaks=2, delta=20, batch_size=64,
                 params_override={}, loss_period=5000, save_net=True, **kwargs):
    """
    Runs a given scenario with a trainer.
    """

    is_conveyors = RunnerClass.context == 'conveyors'

    if pkg_num is not None:
        if is_conveyors:
            prefix = 'bags'
        else:
            prefix = 'pkg'

        def _seq_item(num, d):
            return {prefix+'_number': num, 'delta': d}

        seq = [_seq_item(pkg_num, delta)]

        if not is_conveyors:
            # TODO: breaks in conveyors
            for cur_breaks in range(1, max_breaks+1):
                for i in range(breaks_num):
                    for j in range(cur_breaks):
                        seq.append[{'action': 'restore_link', 'random': True, 'pause': 0}]
                    for j in range(cur_breaks):
                        seq.append[{'action': 'break_link', 'random': True, 'pause': delta}]

                    seq.append(_seq_item(pkg_num, delta))

        params_override = dict_merge(params_override, {
            'settings': {prefix+'_distr': {
                'sequence': seq
            }}
        })

    loss_series = event_series(loss_period, ['count', 'sum'])
    params_override = dict_merge(params_override, {
        'settings': {'router': {router_type: {
            'loss_series': loss_series,
            'batch_size': batch_size,
            'mem_capacity': batch_size
        }}}
    })

    launch_data, runner = run_simulation(return_runner=True,
        RunnerClass=RunnerClass, router_type=router_type, ignore_saved=True,
        training_router_type=training_router_type,
        params_override=params_override, **kwargs)

    loss_series.addAvg()
    brain = runner.world.factory.brain
    if save_net:
        brain.save()

    return brain, loss_series.getSeries(), launch_data

