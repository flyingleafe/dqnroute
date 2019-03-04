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
from ...time_env import *
from ...event_series import *
from ..conveyor_env import *

logger = logging.getLogger(DQNROUTE_LOGGER)

def run_conveyor_scenario(run_params, router_type: str, event_series: EventSeries,
                          random_seed = None, progress_step = None, progress_queue = None) -> EventSeries:
    """
    Runs a conveyor network test scenario with given params and
    aggregates test data in a given `EventSeries`.
    """
    pass


