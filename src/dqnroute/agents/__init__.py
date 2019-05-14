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
}

_conveyors_router_classes = {
    'simple_q': SimpleQRouterConveyor,
    'pred_q': PredictiveQRouterConveyor,
    'link_state': LinkStateRouterConveyor,
    # 'glob_dyn': GlobalDynamicRouterConveyor,
    'dqn': DQNRouterConveyor,
    'dqn_oneout': DQNRouterOOConveyor,
    'dqn_emb': DQNRouterEmbConveyor,
}

def get_router_class(router_type: str, context: Optional[str] = None):
    try:
        if context is None:
            raise Exception('Simulation context is not provided: '\
                            'should be "network" or "conveyors"')
        elif context == 'network':
            return _network_router_classes[router_type]
        elif context == 'conveyors':
            return _conveyors_router_classes[router_type]
        else:
            raise Exception('Unknown simulation context "{}" '\
                            '(should be "network" or "conveyors")'.format(context))
    except KeyError:
        raise UnsupportedRouterType(router_type)
