"""
Runs a conveyor network simulation
"""

import networkx as nx
import pandas as pd

from typing import Tuple, Dict, List
from simpy import Environment

from ...agents import *
from ...messages import *
from ...constants import *
from ...utils import *
from ...event_series import *
from ..conveyor_env import *
from .common import *

logger = logging.getLogger(DQNROUTE_LOGGER)

def _run_scenario(env: Environment, sections_map: Dict[int, SimpyConveyorEnv],
                  sources: List[int], sinks: List[int],
                  bag_distr, random_seed=None):
    if random_seed is not None:
        random.seed(random_seed)

    bag_id = 1
    for period in bag_distr['sequence']:
        delta = period['delta']
        cur_sources = period.get('sources', sources)
        cur_sinks = period.get('sinks', sinks)

        for i in range(0, period['bags_number']):
            src = random.choice(cur_sources)
            dst = random.choice(cur_sinks)
            bag = Bag(bag_id, dst, env.now, None)
            logger.debug("Sending random bag #{} from {} to {} at time {}"
                         .format(bag_id, src, dst, env.now))
            sections_map[src].handle(InMessage(-1, src, PkgMessage(bag)))
            bag_id += 1
            yield env.timeout(delta)

def _get_router_class(router_type):
    if router_type == "simple_q":
        return SimpleQRouterConveyor
    elif router_type == "link_state":
        return LinkStateRouter
    elif router_type == "dqn":
        return DQNRouterConveyor
    else:
        raise Exception("Unsupported router type: " + router_type)

def run_conveyor_scenario(run_params, router_type: str,
                          time_series: EventSeries, energy_series: EventSeries,
                          random_seed = None, progress_step = None,
                          progress_queue = None) -> Tuple[EventSeries, EventSeries]:
    """
    Runs a conveyor network test scenario with given params and
    aggregates test data in a given `EventSeries`.
    """

    configuration = run_params['configuration']
    layout = configuration['layout']
    sources = configuration['sources']
    sinks = configuration['sinks']
    G = make_conveyor_graph(layout)

    env = Environment()
    sections_map = {}
    conveyors = []
    ChosenRouter = _get_router_class(router_type)

    for (i, conveyor) in enumerate(layout):
        routers_args = {}
        for sec_id in conveyor.keys():
            args = make_router_cfg(ChosenRouter, sec_id, G, run_params)
            args['edge_weight'] = 'length'
            routers_args[sec_id] = args

        conveyor_env = SimpyConveyorEnv(env, ChosenRouter, i, conveyor,
                                        time_series=time_series, energy_series=energy_series,
                                        routers_init_args=routers_args,
                                        **run_params['settings']['conveyor_env'])

        for sec_id in conveyor.keys():
            sections_map[sec_id] = conveyor_env
        conveyors.append(conveyor_env)

    for conveyor in conveyors:
        conveyor.init(sections_map, {})

    logger.setLevel(logging.DEBUG)
    env.process(_run_scenario(env, sections_map, sources, sinks,
                              run_params['settings']['bags_distr'], random_seed))

    run_env_progress(env, router_type=router_type, random_seed=random_seed,
                     progress_step=progress_step, progress_queue=progress_queue)

    return time_series, energy_series
