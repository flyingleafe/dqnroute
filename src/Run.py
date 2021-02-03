import os
import argparse
import yaml
import re

import hashlib
import base64

from pathlib import Path
from tqdm import tqdm
from typing import *

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torch

from dqnroute.constants import TORCH_MODELS_DIR
from dqnroute.event_series import split_dataframe
from dqnroute.generator import gen_episodes
from dqnroute.networks.common import get_optimizer
from dqnroute.networks.embeddings import Embedding, LaplacianEigenmap
from dqnroute.networks.q_network import QNetwork
from dqnroute.simulation.common import mk_job_id, add_cols, DummyProgressbarQueue
from dqnroute.simulation.conveyors import ConveyorsRunner
from dqnroute.utils import AgentId, get_amatrix_cols, make_batches, stack_batch

from dqnroute.verification.ml_util import Util
from dqnroute.verification.router_graph import RouterGraph
from dqnroute.verification.adversarial import PGDAdversary
from dqnroute.verification.markov_analyzer import MarkovAnalyzer
from dqnroute.verification.symbolic_analyzer import SymbolicAnalyzer, LipschitzBoundComputer
from dqnroute.verification.nnet_verifier import NNetVerifier, marabou_float2str
from dqnroute.verification.embedding_packer import EmbeddingPacker

NETWORK_FILENAME = "../network.nnet"
PROPERTY_FILENAME = "../property.txt"

parser = argparse.ArgumentParser(
    description="Script to train, simulate and verify deep neural networks for baggage routing.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

# general parameters
parser.add_argument("config_files", type=str, nargs="+",
                    help="YAML config file(s) with the conveyor topology graph, input scenario and settings "
                         "of routing algorithms (all files will be concatenated into one)")
parser.add_argument("--routing_algorithms", type=str, default="dqn_emb,centralized_simple,link_state,simple_q",
                    help="comma-separated list of routing algorithms to run (possible entries: "
                         "dqn_emb, centralized_simple, link_state, simple_q)")
parser.add_argument("--command", type=str, default="run",
                    help="one of run, compute_expected_cost, embedding_adversarial_search, "
                         "embedding_adversarial_verification, q_adversarial_search, q_adversarial_verification")
parser.add_argument("--random_seed", type=int, default=42,
                    help="random seed for pretraining and training")
parser.add_argument("--pretrain_num_episodes", type=int, default=10000,
                    help="number of episodes for supervised pretraining")
parser.add_argument("--force_pretrain", action="store_true",
                    help="whether not to load previously saved pretrained models and force recomputation")
parser.add_argument("--force_train", action="store_true",
                    help="whether not to load previously saved trained models and force recomputation")
parser.add_argument("--skip_graphviz", action="store_true",
                    help="do not visualize graphs with Graphviz")

# common verification / adversarial search parameters
parser.add_argument("--cost_bound", type=float, default=100.0,
                    help="upper bound on expected delivery cost to verify")
parser.add_argument("--simple_path_cost", action="store_true",
                    help="use the number of transitions instead of the total conveyor length as path cost")
parser.add_argument("--input_eps_l_inf", type=float, default=0.1,
                    help="maximum L_∞ discrepancy of input embeddings in adversarial robustness "
                         "verification or search (default: 0.1)")
parser.add_argument("--single_source", type=int, default=None,
                    help="index of the single source to consider (if not specified, all sources will "
                         "be considered)")
parser.add_argument("--single_sink", type=int, default=None,
                    help="index of the single sink to consider (if not specified, all sinks will "
                         "be considered)")
parser.add_argument("--learning_step_indices", type=str, default=None,
                    help="in learning step verification, consider only learning steps with these indices "
                         "comma-separated list without spaces; all steps will be considered if not specified)")

# parameters specific to adversarial search with PGD (embedding_adversarial_search)
parser.add_argument("--input_eps_l_2", type=float, default=1.5,
                    help="maximum (scaled by dimension) L_2 discrepancy of input embeddings in "
                         "adversarial search")
parser.add_argument("--adversarial_search_use_l_2", action="store_true",
                    help="use L_2 norm (scaled by dimension) instead of L_∞ norm during adversarial search")

# parameters specific to learning step verification
# (q_adversarial_search, q_adversarial_verification)
parser.add_argument("--verification_lr", type=float, default=0.001,
                    help="learning rate in learning step verification")
parser.add_argument("--input_max_delta_q", type=float, default=10.0,
                    help="maximum ΔQ in learning step verification")
parser.add_argument("--q_adversarial_no_points", type=int, default=351,
                    help="number of points used to create plots in command q_adversarial")
parser.add_argument("--q_adversarial_verification_no_points", type=int, default=351,
                    help="number of points to search for counterexamples before estimating the Lipschitz "
                         "constant in command q_adversarial_lipschitz (setting to less than 2 disables "
                         "this search)")

# parameters specific to verification with Marabou
# (embedding_adversarial_verification, embedding_adversarial_full_verification)
parser.add_argument("--marabou_path", type=str, default=None,
                    help="path to the Marabou executable")
parser.add_argument("--linux_marabou_memory_limit_mb", type=int, default=None,
                    help="set a memory limit in MB for Marabou (use only on Linux; default: no limit)")

args = parser.parse_args()

# dqn_emb = DQNroute-LE, centralized_simple = BSR
assert len(args.routing_algorithms) > 0, "--routing_algorithms cannot be empty"
router_types = re.split(", *", args.routing_algorithms)
router_types_supported = "dqn_emb centralized_simple link_state simple_q".split(" ")
assert len(set(router_types) - set(router_types_supported)) == 0,\
    f"unsupported algorithm in --routing_algorithms was found; supported ones: {router_types_supported}"
if "dqn_emb" in router_types:
    # always process dqn_emb first
    router_types.remove("dqn_emb")
    router_types.insert(0, "dqn_emb")
nn_loading_needed = router_types[0] == "dqn_emb" or args.command != "run"

for dirname in ["../logs", "../img"]:
    os.makedirs(dirname, exist_ok=True)

    
# 1. load scenario from one or more config files
string_scenario, filename_suffix = [], []
for config_filename in args.config_files:
    filename_suffix += [os.path.split(config_filename)[1].replace(".yaml", "")]
    with open(config_filename, "r") as f:
        string_scenario += f.readlines()
string_scenario = "".join(string_scenario)
scenario = yaml.safe_load(string_scenario)
print(f"Configuration files: {args.config_files}")

router_settings = scenario["settings"]["router"]
emb_dim = router_settings["dqn_emb"]["embedding"]["dim"]
softmax_temperature = router_settings["dqn"]["softmax_temperature"]
probability_smoothing = router_settings["dqn"]["probability_smoothing"]

# graphs size = #sources + #diverters + #sinks + #(conveyors leading to other conveyors)
lengths = [len(scenario["configuration"][x]) for x in ["sources", "diverters", "sinks"]] \
    + [len([c for c in scenario["configuration"]["conveyors"].values()
           if c["upstream"]["type"] == "conveyor"])]
graph_size = sum(lengths)
filename_suffix = "__".join(filename_suffix)
filename_suffix = f"_{emb_dim}_{graph_size}_{filename_suffix}.bin"
print(f"Embedding dimension: {emb_dim}, graph size: {graph_size}")

# 2. supervised pretraining (only for dqn_emb)

def pretrain(args, dir_with_models: str, pretrain_filename: str):
    """ Almost copied from the pretraining notebook. """
    
    def gen_episodes_progress(num_episodes, **kwargs):
        with tqdm(total=num_episodes) as bar:
            return gen_episodes(bar=bar, num_episodes=num_episodes, **kwargs)
    
    class CachedEmbedding(Embedding):
        def __init__(self, InnerEmbedding, dim, **kwargs):
            self.dim = dim
            self.InnerEmbedding = InnerEmbedding
            self.inner_kwargs = kwargs
            self.fit_embeddings = {}

        def fit(self, graph, **kwargs):
            h = hash_graph(graph)
            if h not in self.fit_embeddings:
                embed = self.InnerEmbedding(dim=self.dim, **self.inner_kwargs)
                embed.fit(graph, **kwargs)
                self.fit_embeddings[h] = embed

        def transform(self, graph, idx):
            h = hash_graph(graph)
            return self.fit_embeddings[h].transform(idx)
    
    def shuffle(df):
        return df.reindex(np.random.permutation(df.index))
    
    def hash_graph(graph):
        if type(graph) != np.ndarray:
            graph = nx.to_numpy_matrix(graph, nodelist=sorted(graph.nodes))
        m = hashlib.sha256()
        m.update(graph.tobytes())
        return base64.b64encode(m.digest()).decode("utf-8")
    
    def add_inp_cols(tag, dim):
        return mk_num_list(tag + "_", dim) if dim > 1 else tag

    def qnetwork_batches(net, data, batch_size=64, embedding=None):
        n = net.graph_size
        data_cols = []
        amatrix_cols = get_amatrix_cols(n)
        for tag, dim in net.add_inputs:
            data_cols.append(amatrix_cols if tag == "amatrix" else add_inp_cols(tag, dim))
        for a, b in make_batches(data.shape[0], batch_size):
            batch = data[a:b]
            addr = batch["addr"].values
            dst = batch["dst"].values
            nbr = batch["neighbour"].values
            if embedding is not None:
                amatrices = batch[amatrix_cols].values
                new_btch = []
                for addr_, dst_, nbr_, A in zip(addr, dst, nbr, amatrices):
                    A = A.reshape(n, n)
                    embedding.fit(A)
                    new_addr = embedding.transform(A, int(addr_))
                    new_dst = embedding.transform(A, int(dst_))
                    new_nbr = embedding.transform(A, int(nbr_))
                    new_btch.append((new_addr, new_dst, new_nbr))
                [addr, dst, nbr] = stack_batch(new_btch)
            addr_inp = torch.FloatTensor(addr)
            dst_inp = torch.FloatTensor(dst)
            nbr_inp = torch.FloatTensor(nbr)
            inputs = tuple(torch.FloatTensor(batch[cols].values) for cols in data_cols)
            output = torch.FloatTensor(batch["predict"].values)
            yield (addr_inp, dst_inp, nbr_inp) + inputs, output

    def qnetwork_pretrain_epoch(net, optimizer, data, **kwargs):
        loss_func = torch.nn.MSELoss()
        for batch, target in qnetwork_batches(net, data, **kwargs):
            optimizer.zero_grad()
            output = net(*batch)
            loss = loss_func(output, target.unsqueeze(1))
            loss.backward()
            optimizer.step()
            yield float(loss)

    def qnetwork_pretrain(net, data, optimizer="rmsprop", epochs=1, save_net=True, **kwargs):
        optimizer = get_optimizer(optimizer)(net.parameters())
        epochs_losses = []
        for i in tqdm(range(epochs)):
            sum_loss = 0
            loss_cnt = 0
            for loss in tqdm(qnetwork_pretrain_epoch(net, optimizer, data, **kwargs), desc=f"epoch {i}"):
                sum_loss += loss
                loss_cnt += 1
            epochs_losses.append(sum_loss / loss_cnt)
        if save_net:
            # label changed by Igor:
            net._label = pretrain_filename
            net.save()
        return epochs_losses
    
    data_conv = gen_episodes_progress(ignore_saved=True, context="conveyors",
                                      num_episodes=args.pretrain_num_episodes,
                                      random_seed=args.random_seed, run_params=scenario,
                                      save_path="../logs/data_conveyor1_oneinp_new.csv")
    data_conv.loc[:, "working"] = 1.0
    conv_emb = CachedEmbedding(LaplacianEigenmap, dim=emb_dim)
    args = {
        "scope": dir_with_models,
        "activation": router_settings["dqn"]["activation"],
        "layers": router_settings["dqn"]["layers"],
        "embedding_dim": emb_dim,
    }
    conveyor_network_ng_emb = QNetwork(graph_size, **args)
    conveyor_network_ng_emb_losses = qnetwork_pretrain(conveyor_network_ng_emb, shuffle(data_conv), epochs=10,
                                                       embedding=conv_emb, save_net=True)
    # conveyor_network_ng_emb_ws = QNetwork(graph_size, additional_inputs=[{"tag": "working", "dim": 1}], **args)
    # conveyor_network_ng_emb_ws_losses = qnetwork_pretrain(conveyor_network_ng_emb_ws, shuffle(data_conv), epochs=20,
    #                                                       embedding=conv_emb, save_net=False)

dir_with_models = "conveyor_test_ng"
pretrain_filename = f"pretrained{filename_suffix}"
pretrain_path = Path(TORCH_MODELS_DIR) / dir_with_models / pretrain_filename
do_pretrain = args.force_pretrain or not pretrain_path.exists()
if nn_loading_needed:
    if do_pretrain:
        print(f"Pretraining {pretrain_path}...")
        pretrain(args, dir_with_models, pretrain_filename)
    else:
        print(f"Using the already pretrained model {pretrain_path}...")


# 3. train (only for dqn_emb)

# TODO check whether setting a random seed makes training deterministic

def run_single(run_params: dict, router_type: str, random_seed: int, **kwargs):
    job_id = mk_job_id(router_type, random_seed)
    with tqdm(desc=job_id) as bar:
        queue = DummyProgressbarQueue(bar)
        runner = ConveyorsRunner(run_params=run_params, router_type=router_type, random_seed=random_seed,
                                 progress_queue=queue, **kwargs)
        event_series = runner.run(**kwargs)
    return event_series, runner

def train(args, dir_with_models: str, pretrain_filename: str, train_filename: str,
          router_type: str, retrain: bool, work_with_files: bool):
    # added by Igor (TODO implement in a proper way):
    os.environ["IGOR_OVERRIDDEN_DQN_LOAD_FILENAME"] = pretrain_filename
    if retrain:
        if "IGOR_OMIT_TRAINING" in os.environ:
            del os.environ["IGOR_OMIT_TRAINING"]
    else:
        os.environ["IGOR_OMIT_TRAINING"] = "True"
    
    event_series, runner = run_single(run_params=scenario, router_type=router_type, progress_step=500,
                                      ignore_saved=[True], random_seed=args.random_seed)
    if router_type == "dqn_emb":
        world = runner.world
        net = next(iter(next(iter(world.handlers.values())).routers.values())).brain
        net._label = train_filename    
        # save or load the trained network
        if work_with_files:
            if retrain:
                net.save()
            else:
                net.restore()
    else:
        world = None
    return event_series, world
    

train_filename = f"trained{filename_suffix}"
train_path = Path(TORCH_MODELS_DIR) / dir_with_models / train_filename
do_train = args.force_train or not train_path.exists() or args.command == "run" or do_pretrain
if nn_loading_needed:
    if do_train:
        print(f"Training {train_path}...")
    else:
        print(f"Using the already trained model {train_path}...")
    dqn_log, world = train(args, dir_with_models, pretrain_filename, train_filename, "dqn_emb", do_train, True)


# 4. load the router graph
if nn_loading_needed:
    g = RouterGraph(world)
    print("Reachability matrix:")
    g.print_reachability_matrix()

def visualize(g: RouterGraph):
    gv_graph = g.to_graphviz()
    prefix = f"../img/topology_graph{filename_suffix}."
    gv_graph.write(prefix + "gv")
    for prog in ["dot", "circo", "twopi"]:
        prog_prefix = f"{prefix}{prog}."
        for fmt in ["pdf", "png"]:
            path = f"{prog_prefix}{fmt}"
            print(f"Drawing {path} ...")
            gv_graph.draw(path, prog=prog, args="-Gdpi=300 -Gmargin=0 -Grankdir=LR")

if nn_loading_needed and not args.skip_graphviz:
    visualize(g)

def get_symbolic_analyzer() -> SymbolicAnalyzer:
    return SymbolicAnalyzer(g, softmax_temperature, probability_smoothing,
                            args.verification_lr, delta_q_max=args.input_max_delta_q)

def get_nnet_verifier() -> NNetVerifier:
    assert args.marabou_path is not None,\
        "You must specify --verification_marabou_path for command embedding_adversarial_verification."
    return NNetVerifier(g, args.marabou_path, NETWORK_FILENAME, PROPERTY_FILENAME, probability_smoothing,
                        softmax_temperature, emb_dim, args.linux_marabou_memory_limit_mb)

def get_sources(sink: AgentId) -> List[AgentId]:
    """
    :return: a list of all sources that are reachable from the specified sink. If a single source
        was specified in command line arguments, only this source will be returned.
    """
    return [source for source in g.get_sources_for_node(sink)
            if args.single_source is None or source[1] == args.single_source]

def get_sinks() -> List[Tuple[AgentId, torch.Tensor]]:
    """
    :return: a list of all sinks. If a single sink was specified in command line arguments, only
        this sink will be returned.
    """
    return [(sink, g.node_to_embeddings(sink, sink)[0]) for sink in g.sinks
            if args.single_sink is None or sink[1] == args.single_sink]

def get_source_sink_pairs(message: str, ma_verbose: bool = False) -> \
        Generator[Tuple[AgentId, AgentId, torch.Tensor, MarkovAnalyzer], None, None]:
    for sink, sink_embedding in get_sinks():
        sink_embeddings = sink_embedding.repeat(2, 1)
        for source in get_sources(sink):
            print(f"{message} from {source} to {sink}...")
            ma = MarkovAnalyzer(g, source, sink, args.simple_path_cost, ma_verbose)
            yield source, sink, sink_embedding, ma 
    
def get_learning_step_indices() -> Optional[Set[int]]:
    if args.learning_step_indices is None:
        return None
    return set([int(s) for s in args.learning_step_indices.split(",")])

print(f"Running command {args.command}...")

# Simulate and make plots
if args.command == "run":
    _legend_txt_replace = {
        "networks": {
            "link_state": "Shortest paths", "simple_q": "Q-routing", "pred_q": "PQ-routing",
            "glob_dyn": "Global-dynamic", "dqn": "DQN", "dqn_oneout": "DQN (1-out)",
            "dqn_emb": "DQN-LE", "centralized_simple": "Centralized control"
        }, "conveyors": {
            "link_state": "Vyatkin-Black", "simple_q": "Q-routing", "pred_q": "PQ-routing",
            "glob_dyn": "Global-dynamic", "dqn": "DQN", "dqn_oneout": "DQN (1-out)",
            "dqn_emb": "DQN-LE", "centralized_simple": "BSR"
        }
    }
    _targets = {"time": "avg", "energy": "sum", "collisions": "sum"}
    _ylabels = {
        "time": "Mean delivery time",
        "energy": "Total energy consumption",
        "collisions": "Cargo collisions"
    }
    
    
    series = []
    # reuse the log for dqn_emb:
    if router_types[0] == "dqn_emb":
        series += [dqn_log.getSeries(add_avg=True)]
    # perform training/simulation with other approaches
    for router_type in router_types:
        if router_type != "dqn_emb":
            s, _ = train(args, dir_with_models, pretrain_filename, train_filename, router_type, True, False)
            series += [s.getSeries(add_avg=True)]
    
    dfs = []
    for router_type, s in zip(router_types, series):
        df = s.copy()
        add_cols(df, router_type=router_type, seed=args.random_seed)
        dfs.append(df)
    dfs = pd.concat(dfs, axis=0)
    
    def print_sums(df):
        for tp in router_types:
            x = df.loc[df["router_type"] == tp, "count"].sum()
            txt = _legend_txt_replace.get(tp, tp)
            print(f"  {txt}: {x}")
    
    def plot_data(data, meaning="time", figsize=(15,5), xlim=None, ylim=None,
              xlabel="Simulation time", ylabel=None, font_size=14, title=None, save_path=None,
              draw_collisions=False, context="networks", **kwargs):
        if "time" not in data.columns:
            datas = split_dataframe(data, preserved_cols=["router_type", "seed"])
            for tag, df in datas:
                if tag == "collisions" and not draw_collisions:
                    print("Number of collisions:")
                    print_sums(df)
                    continue
                xlim = kwargs.get(tag + "_xlim", xlim)
                ylim = kwargs.get(tag + "_ylim", ylim)
                save_path = kwargs.get(tag + "_save_path", save_path)
                plot_data(df, meaning=tag, figsize=figsize, xlim=xlim, ylim=ylim,
                          xlabel=xlabel, ylabel=ylabel, font_size=font_size,
                          title=title, save_path=save_path, context="conveyors")
            return

        target = _targets[meaning]
        if ylabel is None:
            ylabel = _ylabels[meaning]

        fig = plt.figure(figsize=figsize)
        ax = sns.lineplot(x="time", y=target, hue="router_type", data=data, err_kws={"alpha": 0.1})
        handles, labels = ax.get_legend_handles_labels()
        new_labels = list(map(lambda l: _legend_txt_replace[context].get(l, l), labels[1:]))
        ax.legend(handles=handles[1:], labels=new_labels, fontsize=font_size)
        ax.tick_params(axis="both", which="both", labelsize=int(font_size*0.75))
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
        if title is not None:
            ax.set_title(title)
        ax.set_xlabel(xlabel, fontsize=font_size)
        ax.set_ylabel(ylabel, fontsize=font_size)

        if save_path is not None:
            fig.savefig(f"../img/{save_path}", bbox_inches="tight")
    
    plot_data(dfs, figsize=(14, 8), font_size=22,
              time_save_path="time-plot.pdf", energy_save_path="energy-plot.pdf")

# Compute the expression of the expected delivery cost and evaluate it
elif args.command == "compute_expected_cost":
    sa = get_symbolic_analyzer()
    for source, sink, sink_embedding, ma in get_source_sink_pairs("Delivery", True):
        _, lambdified_objective = ma.get_objective()
        ps = sa.compute_ps(ma, sink, sink_embedding.repeat(2, 1), 0, 0)
        objective_value = lambdified_objective(*ps)
        print(f"  Computed probabilities: {Util.list_round(ps, 6)}")
        print(f"  E(delivery cost from {source} to {sink}) = {objective_value}")
                          
# Search for adversarial examples w.r.t. input embeddings
elif args.command == "embedding_adversarial_search":
    if args.adversarial_search_use_l_2:
        norm, norm_bound = "scaled_l_2", args.input_eps_l_2
    else:
        norm, norm_bound = "l_inf", args.input_eps_l_inf
    adv = PGDAdversary(rho=norm_bound, steps=100, step_size=0.02, random_start=True, stop_loss=args.cost_bound,
                       verbose=2, norm=norm, n_repeat=10, repeat_mode="any", dtype=torch.float64)
    print(f"Trying to falsify ({norm}_norm(Δembedding) ≤ {norm_bound}) => (E(cost) < {args.cost_bound}).")
    for source, sink, sink_embedding, ma in get_source_sink_pairs("Measuring adversarial robustness of delivery"):
        # gather all embeddings that we need to compute the objective
        embedding_packer = EmbeddingPacker(g, sink, sink_embedding, ma.reachable_nodes)
        _, lambdified_objective = ma.get_objective()

        def get_gradient(x: torch.Tensor) -> Tuple[torch.Tensor, float, str]:
            """
            :param x: parameter vector (the one expected to converge to an adversarial example)
            :return: a tuple (gradient pointing to the direction of the adversarial attack,
                              the corresponding loss function value,
                              auxiliary information for printing during optimization).
            """
            x = Util.optimizable_clone(x.flatten())
            objective_value, objective_inputs = embedding_packer.compute_objective(
                embedding_packer.unpack(x), ma.nontrivial_diverters, lambdified_objective,
                softmax_temperature, probability_smoothing)
            objective_value.backward()
            aux_info = ", ".join([f"{param}={value.detach().cpu().item():.4f}"
                                  for param, value in zip(ma.params, objective_inputs)])
            return x.grad, objective_value.item(), f"[{aux_info}]"
            
        best_embedding = adv.perturb(embedding_packer.initial_vector(), get_gradient)
        _, objective, aux_info = get_gradient(best_embedding)
        print("  Found counterexample!" if objective >= args.cost_bound else "  Verified.")
        print(f"  Best perturbed vector: {Util.to_numpy(best_embedding).round(3).flatten().tolist()} {aux_info}")
        print(f"  Best objective value: {objective}")
                          
# Formally verify the expected cost bound w.r.t. input embeddings
elif args.command == "embedding_adversarial_verification":
    nv = get_nnet_verifier()
    for source, sink, _, ma in get_source_sink_pairs("Verifying adversarial robustness of delivery"):
        result = nv.verify_delivery_cost_bound(source, sink, ma, args.input_eps_l_inf, args.cost_bound)
        print(f"  {result}")

# Evaluate the expected delivery cost assuming a change in NN parameters and make plots            
elif args.command == "q_adversarial_search":
    sa = get_symbolic_analyzer()
    learning_step_index = -1
    requested_indices = get_learning_step_indices()
    for source, sink, sink_embedding, ma in get_source_sink_pairs("Measuring robustness of delivery"):
        sink_embeddings = sink_embedding.repeat(2, 1)
        objective, lambdified_objective = ma.get_objective()
        for node_key in g.node_keys:
            current_embedding, neighbors, neighbor_embeddings = g.node_to_embeddings(node_key, sink)
            for neighbor_key, neighbor_embedding in zip(neighbors, neighbor_embeddings):
                learning_step_index += 1
                if requested_indices is not None and learning_step_index not in requested_indices:
                    continue
                print(f"  Considering learning step {node_key} → {neighbor_key}...")
                # compute
                # we assume a linear change of parameters
                reference_q = sa.compute_gradients(current_embedding, sink_embedding,
                                                    neighbor_embedding).flatten().item()
                actual_qs = np.linspace(-sa.delta_q_max, sa.delta_q_max,
                                        args.q_adversarial_no_points) + reference_q
                kappa, lambdified_kappa = sa.get_transformed_cost(ma, objective, args.cost_bound)
                objective_values, kappa_values = [torch.empty(len(actual_qs)) for _ in range(2)]
                for i, actual_q in enumerate(actual_qs):
                    ps = sa.compute_ps(ma, sink, sink_embeddings, reference_q, actual_q)
                    objective_values[i] = lambdified_objective(*ps)
                    kappa_values[i]     = lambdified_kappa(*ps)
                #print(((objective_values > args.cost_bound) != (kappa_values > 0)).sum()) 
                fig, axes = plt.subplots(3, 1, figsize=(10, 10))
                plt.subplots_adjust(hspace=0.4)
                caption_starts = *(["Delivery cost (τ)"] * 2), "Transformed delivery cost (κ)"
                values         = *([objective_values   ] * 2), kappa_values
                axes[0].set_yscale("log")
                for ax, caption_start, values in zip(axes, caption_starts, values):
                    label = (f"{caption_start} from {source} to {sink} when making optimization"
                                f" step with current={node_key}, neighbor={neighbor_key}")
                    print(f"    Plotting: {caption_start}...")
                    ax.set_title(label)
                    ax.plot(actual_qs, values)
                    y_delta = 0 if np.ptp(values) > 0 else 5
                    # show the zero step value:
                    ax.vlines(reference_q, min(values) - y_delta, max(values) + y_delta)
                    ax.hlines(values[len(values) // 2], min(actual_qs), max(actual_qs))
                # show the verification bound:
                for i in range(2):
                    axes[i].hlines(args.cost_bound, min(actual_qs), max(actual_qs))
                axes[2].hlines(0, min(actual_qs), max(actual_qs))
                plt.savefig(f"../img/{filename_suffix}_{learning_step_index}.pdf", bbox_inches = "tight")
                plt.close()
                print(f"    Empirically found maximum of τ: {objective_values.max():.6f}")
                print(f"    Empirically found maximum of κ: {kappa_values.max():.6f}")

# Formally verify the bound on the expected delivery cost w.r.t. learning step magnitude  
elif args.command == "q_adversarial_verification":
    sa = get_symbolic_analyzer()
    sa.load_matrices()
    learning_step_index = -1
    requested_indices = get_learning_step_indices()
    for source, sink, sink_embedding, ma in get_source_sink_pairs("Verifying robustness of delivery"):
        objective, _ = ma.get_objective()
        for node_key in g.node_keys:
            current_embedding, neighbors, neighbor_embeddings = g.node_to_embeddings(node_key, sink)
            for neighbor_key, neighbor_embedding in zip(neighbors, neighbor_embeddings):
                learning_step_index += 1
                if requested_indices is not None and learning_step_index not in requested_indices:
                    continue
                print(f"  Considering learning step {node_key} → {neighbor_key}...")
                lbc = LipschitzBoundComputer(sa, ma, objective, sink, current_embedding,
                                             sink_embedding, neighbor_embedding, args.cost_bound)
                if lbc.prove_bound(args.q_adversarial_verification_no_points):
                    print("    Proof found!")
                print(f"    Number of evaluations of κ: {lbc.no_evaluations}")
                print(f"    Maximum depth reached: {lbc.max_depth}")

else:
    raise RuntimeError(f"Unknown command {args.command}.")
