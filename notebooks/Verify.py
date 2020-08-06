import numpy as np
import torch
import matplotlib.pyplot as plt
import argparse
import yaml

from tqdm import tqdm
from typing import *

import os
current_dir = os.getcwd()
os.chdir("../src")
from dqnroute import *
os.chdir(current_dir)

parser = argparse.ArgumentParser(description="Verifier of baggage routing neural networks.")
parser.add_argument("--command", type=str, required=True,
                    help="one of deterministic_test, embedding_adversarial, q_adversarial, compare")
parser.add_argument("--config_file", type=str, required=True,
                    help="YAML config file with the topology graph and other configuration info")
parser.add_argument("--smoothing", type=float, default=0.01,
                    help="smoothing (0..1) of probabilities during learning and verification (defaut: 0.01)")
parser.add_argument("--random_seed", type=int, default=42,
                    help="random seed for pretraining and training (default: 42)")
parser.add_argument("--force_pretrain", action="store_true",
                    help="whether not to load previously saved pretrained models and force recomputation")
parser.add_argument("--force_train", action="store_true",
                    help="whether not to load previously saved trained models and force recomputation")

args = parser.parse_args()

def run_single(file: str, router_type: str, random_seed: int, **kwargs):
    job_id = mk_job_id(router_type, random_seed)
    with tqdm(desc=job_id) as bar:
        queue = DummyProgressbarQueue(bar)
        runner = ConveyorsRunner(run_params=file, router_type=router_type,
                                 random_seed=random_seed, progress_queue=queue, **kwargs)
        event_series = runner.run(**kwargs)
    return event_series, runner

# 1. pretrain

def pretrain(args):
    # TODO read yaml and get parameters
    scenario = args.config_file
    
    with open(scenario) as file:
        scenario_loaded = yaml.load(file, Loader=yaml.FullLoader)
    
    emb_dim = scenario_loaded["settings"]["router"]["dqn_emb"]["embedding"]["dim"]
    print(emb_dim)
    
    # graphs size = #sources + #diverters + #sinks + #(conveyors leading to other conveyors)
    lengths = [len(scenario_loaded["configuration"][x]) for x in ["sources", "diverters", "sinks"]] \
        + [len([c for c in scenario_loaded["configuration"]["conveyors"].values()
               if c["upstream"]["type"] == "conveyor"])]
    #print(lens)
    graph_size = sum(lengths)
    #print(graph_size)
    # FIXME wrong formula!
    
    data_conv = gen_episodes_progress(ignore_saved=True,
        context='conveyors', num_episodes=10000, random_seed=args.random_seed,
        run_params=scenario,
        save_path='../logs/data_conveyor1_oneinp_new.csv')
    # TODO get graph size with more clever means
    
# TODO save/load
pretrain(args)    


if args.command == "deterministic_test":
    pass
elif args.command == "embedding_adversarial":
    pass
elif args.command == "q_adversarial":
    pass
elif args.command == "compare":
    # TODO compare the trained model with Vyatkin/black
    pass
else:
    raise RuntimeError(f"Unknown command {args.command}.")