"""
Runs a computer network simulation
"""

import networkx as nx
import pandas as pd
import tensorflow as tf

from typing import Dict
from simpy import Environment
from copy import deepcopy

from ...agents import *
from ...messages import *
from ...constants import *
from ...utils import *
from ...event_series import *
from ..router_env import *
from .common import *

logger = logging.getLogger(DQNROUTE_LOGGER)

def _run_scenario(env: Environment, routers: Dict[str, SimpyRouterEnv],
                  G: nx.DiGraph, pkg_distr, random_seed = None):
    if random_seed is not None:
        random.seed(random_seed)

    pkg_id = 1
    for period in pkg_distr["sequence"]:
        try:
            action = period["action"]
            pause = period["pause"]
            u = period["u"]
            v = period["v"]

            if action == 'break_link':
                routers[u].handle(RemoveLinkMessage(v))
                routers[v].handle(RemoveLinkMessage(u))
            elif action == 'restore_link':
                routers[u].handle(AddLinkMessage(v, G.edges[u, v]))
                routers[v].handle(AddLinkMessage(u, G.edges[v, u]))
            yield env.timeout(pause)
        except KeyError:
            delta = period["delta"]
            all_nodes = list(routers.keys())
            sources = period.get("sources", all_nodes)
            dests = period.get("dests", all_nodes)

            for i in range(0, period["pkg_number"]):
                src = random.choice(sources)
                dst = random.choice(dests)
                pkg = Package(pkg_id, 1024, dst, env.now, None) # create empty packet
                logger.debug("Sending random pkg #{} from {} to {} at time {}"
                             .format(pkg_id, src, dst, env.now))
                routers[src].handle(InMessage(-1, src, PkgMessage(pkg)))
                pkg_id += 1
                yield env.timeout(delta)

def _get_router_class(router_type):
    if router_type == "simple_q":
        return SimpleQRouterNetwork
    elif router_type == "link_state":
        return LinkStateRouter
    elif router_type == "dqn":
        return DQNRouterNetwork
    else:
        raise Exception("Unsupported router type: " + router_type)

def run_network_scenario(run_params, router_type: str, event_series: EventSeries,
                         random_seed = None, progress_step = None, progress_queue = None) -> EventSeries:
    """
    Runs a computer network test scenario with given params and
    aggregates test data in a given `EventSeries`.
    """

    G = make_network_graph(run_params['network'])

    env = Environment()
    routers = {}
    ChosenRouter = _get_router_class(router_type)

    for node, nbrs in G.adjacency():
        edges = [(node, v, attrs) for v, attrs in nbrs.items()]
        routers[node] = SimpyRouterEnv(env, ChosenRouter,
                                       router_id=node,
                                       router_init_args=make_router_cfg(ChosenRouter, node, G, run_params),
                                       data_series=event_series, edges=edges,
                                       **run_params['settings']['router_env'])

    for node in G.nodes():
        out_routers = {v: routers[v] for (_, v) in G.out_edges(node)}
        routers[node].init(out_routers, {})

    logger.setLevel(logging.DEBUG)
    env.process(_run_scenario(env, routers, G, run_params['settings']['pkg_distr'], random_seed))
    run_env_progress(env, router_type=router_type, random_seed=random_seed,
                     progress_step=progress_step, progress_queue=progress_queue)

    if issubclass(ChosenRouter, DQNRouter):
        for router_env in routers.values():
            router_env.handler.session.close()

    return event_series
