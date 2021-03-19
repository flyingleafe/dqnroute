"""
Contains definitions of `MessageHandler` interface and its implementations.
"""

from .base import *
from .routers import *
from .conveyors import *

from typing import Optional

class UnsupportedRouterType(Exception):
    """
    Exception which is thrown when an unknown router type
    (as a string) is provided somewhere.
    """
    pass

_network_router_classes = {
    'simple_q': SimpleQRouterNetwork,
    'pred_q': PredictiveQRouterNetwork,
    'link_state': LinkStateRouter,
    'glob_dyn': GlobalDynamicRouter,
    'dqn': DQNRouterNetwork,
    'dqn_oneout': DQNRouterOONetwork,
    'dqn_emb': DQNRouterEmbNetwork,
    'ppo_emb': PPORouterEmbNetwork,
    'reinforce_emb': ReinforceNetwork,
}

_conveyors_router_classes = {
    'simple_q': SimpleQRouterConveyor,
    'pred_q': PredictiveQRouterConveyor,
    'link_state': LinkStateRouterConveyor,
    # 'glob_dyn': GlobalDynamicRouterConveyor,
    'dqn': DQNRouterConveyor,
    'dqn_oneout': DQNRouterOOConveyor,
    'centralized_simple': (CentralizedController, CentralizedOracle),
    'dqn_emb': DQNRouterEmbNetwork,
    'ppo_emb': PPORouterEmbConveyor,
    'reinforce_emb': ReinforceConveyor,
}

def get_router_class(router_type: str, context: Optional[str] = None, oracle=False):
    try:
        res = None
        if context is None:
            raise Exception('Simulation context is not provided: '\
                            'should be "network" or "conveyors"')
        elif context == 'network':
            res = _network_router_classes[router_type]
        elif context == 'conveyors':
            res = _conveyors_router_classes[router_type]

        if res is None:
            raise Exception('Unknown simulation context "{}" '\
                            '(should be "network" or "conveyors")'.format(context))
        if type(res) == tuple:
            return res[1] if oracle else res[0]
        return res

    except KeyError:
        raise UnsupportedRouterType(router_type)
