import networkx as nx
import os

from typing import List
from simpy import Environment, Event, Interrupt
from ..event_series import *
from ..messages import *
from ..agents import *
from ..utils import *
from .common import *

def TrainingRouterClass(Trainee: DQNRouter, Trainer: Router, strict_guidance=True):
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

        trainer_estimate = Trainer.pathCost(self, pkg.dst, through=to)
        for msg in trainee_msgs:
            if isinstance(msg, RewardMsg):
                msg.Q_estimate = trainer_estimate

        return to, trainer_msgs + trainee_msgs

    classname = 'Training_{}_{}'.format(Trainee.__name__, Trainer.__name__)
    return type(classname, (Trainee, Trainer), {'route': route, '__init__': __init__})

def run_training_scenario(RunnerClass, router_type, training_router_type, run_params,
                          random_seed=None, loss_period=500, save_net=True, **kwargs):
    loss_series = event_series(loss_period, ['count', 'sum'])
    run_params['settings']['router'][router_type]['loss_series'] = loss_series

    runner = RunnerClass(run_params=run_params, router_type=router_type,
                         training_router_type=training_router_type, **kwargs)
    brain = runner.world.factory.brain

    data_series = runner.run(random_seed=random_seed, ignore_saved=True, **kwargs)
    if save_net:
        brain.save()

    return brain, loss_series, data_series
