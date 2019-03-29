"""
Runs a conveyor network simulation
"""

import networkx as nx
import pandas as pd

from typing import Dict
from simpy import Environment
from copy import deepcopy

from ...agents import *
from ...messages import *
from ...constants import *
from ...utils import *
from ...event_series import *
from ..conveyor_env import *
from .common import *

logger = logging.getLogger(DQNROUTE_LOGGER)

def _get_router_class(router_type):
    if router_type == "simple_q":
        return SimpleQRouterConveyor
    elif router_type == "link_state":
        return LinkStateRouter
    elif router_type == "dqn":
        return DQNRouterConveyor
    else:
        raise Exception("Unsupported router type: " + router_type)

def run_conveyor_scenario(run_params, router_type: str, event_series: EventSeries,
                          random_seed = None, progress_step = None, progress_queue = None) -> EventSeries:
    """
    Runs a conveyor network test scenario with given params and
    aggregates test data in a given `EventSeries`.
    """

    G = make_conveyor_graph(run_params['configuration'])

    env = Environment()
    conveyors = {}
