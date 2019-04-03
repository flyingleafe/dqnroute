"""
Common util functions for runners
"""

from simpy import Environment

from ...agents import *

def make_router_cfg(RouterClass, router_id, G, run_params):
    """
    Makes valid config for the router of a given class
    """

    out_routers = [v for (_, v) in G.out_edges(router_id)]
    in_routers = [v for (v, _) in G.in_edges(router_id)]
    router_cfg = {
        'nodes': list(G.nodes()),
        'out_neighbours': out_routers,
        'in_neighbours': in_routers
    }
    router_cfg.update(run_params['settings']['router'])

    if issubclass(RouterClass, LinkStateRouter):
        router_cfg['network'] = deepcopy(G)
    return router_cfg

def run_env_progress(env: Environment, router_type: str, random_seed = None,
                     progress_step = None, progress_queue = None):
    """
    Runs the environment, optionally reporting the progress to a given queue
    """

    if progress_queue is not None:
        if progress_step is None:
            env.run()
            progress_queue.put((router_type, random_seed, progress_step))
        else:
            next_step = progress_step
            while env.peek() != float('inf'):
                env.run(until=next_step)
                progress_queue.put((router_type, random_seed, progress_step))
                next_step += progress_step
        progress_queue.put((router_type, random_seed, None))
    else:
        env.run()

