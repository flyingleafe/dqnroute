import argparse
import yaml
from pathlib import Path
from tqdm import tqdm
from typing import *
from collections import OrderedDict

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import sympy

import os
current_dir = os.getcwd()
os.chdir("../src")
from dqnroute import *
from dqnroute.networks import *
from dqnroute.verification.router_graph import RouterGraph
from dqnroute.verification.adversarial import PGDAdversary
from dqnroute.verification.ml_util import Util
from dqnroute.verification.markov_analyzer import MarkovAnalyzer
from dqnroute.verification.symbolic_analyzer import SymbolicAnalyzer
from dqnroute.utils import memoize

os.chdir(current_dir)


parser = argparse.ArgumentParser(description="Verifier of baggage routing neural networks.")
parser.add_argument("--command", type=str, required=True,
                    help="one of deterministic_test, embedding_adversarial, q_adversarial, q_adversarial_lipschitz, compare")
parser.add_argument("--config_file", type=str, required=True,
                    help="YAML config file with the topology graph and other configuration info")
parser.add_argument("--probability_smoothing", type=float, default=0.01,
                    help="smoothing (0..1) of probabilities during learning and verification (defaut: 0.01)")
parser.add_argument("--random_seed", type=int, default=42,
                    help="random seed for pretraining and training (default: 42)")
parser.add_argument("--force_pretrain", action="store_true",
                    help="whether not to load previously saved pretrained models and force recomputation")
parser.add_argument("--force_train", action="store_true",
                    help="whether not to load previously saved trained models and force recomputation")
parser.add_argument("--simple_path_cost", action="store_true",
                    help="use the number of transitions instead of the total conveyor length as path cost")
parser.add_argument("--skip_graphviz", action="store_true",
                    help="do not visualize graphs")
parser.add_argument("--softmax_temperature", type=float, default=1.5,
                    help="custom softmax temperature (higher temperature means larger entropy in routing decisions; default: 1.5)")
parser.add_argument("--cost_bound", type=float, default=100.0,
                    help="upper bound on delivery cost to verify (default: 100)")
parser.add_argument("--verification_lr", type=float, default=0.001,
                    help="learning rate in learning step verification (default: 0.001)")
parser.add_argument("--verification_max_delta_q", type=float, default=10.0,
                    help="maximum ΔQ in learning step verification (default: 10.0)")

parser.add_argument("--pretrain_num_episodes", type=int, default=10000,
                    help="pretrain_num_episodes (default: 10000)")

args = parser.parse_args()


os.environ["IGOR_OVERRDIDDED_SOFTMAX_TEMPERATURE"] = str(args.softmax_temperature)

# 1. load scenario
scenario = args.config_file
print(f"Scenario: {scenario}")

with open(scenario) as file:
    scenario_loaded = yaml.load(file, Loader=yaml.FullLoader)

emb_dim = scenario_loaded["settings"]["router"]["dqn_emb"]["embedding"]["dim"]

# graphs size = #sources + #diverters + #sinks + #(conveyors leading to other conveyors)
lengths = [len(scenario_loaded["configuration"][x]) for x in ["sources", "diverters", "sinks"]] \
    + [len([c for c in scenario_loaded["configuration"]["conveyors"].values()
           if c["upstream"]["type"] == "conveyor"])]
#print(lengths)
graph_size = sum(lengths)
print(f"Embedding dimension: {emb_dim}, graph size: {graph_size}")


# 2. pretrain

def pretrain(args, dir_with_models: str, pretrain_filename: str):
    """ ALMOST COPIED FROM THE PRETRAINING NOTEBOOK """
    
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
        return base64.b64encode(m.digest()).decode('utf-8')
    
    def add_inp_cols(tag, dim):
        return mk_num_list(tag + '_', dim) if dim > 1 else tag

    def qnetwork_batches(net, data, batch_size=64, embedding=None):
        n = net.graph_size
        data_cols = []
        amatrix_cols = get_amatrix_cols(n)
        for tag, dim in net.add_inputs:
            data_cols.append(amatrix_cols if tag == 'amatrix' else add_inp_cols(tag, dim))
        for a, b in make_batches(data.shape[0], batch_size):
            batch = data[a:b]
            addr = batch['addr'].values
            dst = batch['dst'].values
            nbr = batch['neighbour'].values
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
            addr_inp = torch.tensor(addr, dtype=torch.float)
            dst_inp = torch.tensor(dst, dtype=torch.float)
            nbr_inp = torch.tensor(nbr, dtype=torch.float)
            inputs = tuple(torch.tensor(batch[cols].values, dtype=torch.float) for cols in data_cols)
            output = torch.tensor(batch['predict'].values, dtype=torch.float)
            yield (addr_inp, dst_inp, nbr_inp) + inputs, output

    def qnetwork_pretrain_epoch(net, optimizer, data, **kwargs):
        loss_func = nn.MSELoss()
        for batch, target in qnetwork_batches(net, data, **kwargs):
            optimizer.zero_grad()
            output = net(*batch)
            loss = loss_func(output, target.unsqueeze(1))
            loss.backward()
            optimizer.step()
            yield float(loss)

    def qnetwork_pretrain(net, data, optimizer='rmsprop', epochs=1, save_net=True, **kwargs):
        optimizer = get_optimizer(optimizer)(net.parameters())
        epochs_losses = []
        for i in tqdm(range(epochs)):
            sum_loss = 0
            loss_cnt = 0
            for loss in tqdm(qnetwork_pretrain_epoch(net, optimizer, data, **kwargs), desc=f'epoch {i}'):
                sum_loss += loss
                loss_cnt += 1
            epochs_losses.append(sum_loss / loss_cnt)
        if save_net:
            # label changed by Igor:
            net._label = pretrain_filename
            net.save()
        return epochs_losses
    
    data_conv = gen_episodes_progress(ignore_saved=True, context='conveyors',
                                      num_episodes=args.pretrain_num_episodes,
                                      random_seed=args.random_seed, run_params=scenario,
                                      save_path='../logs/data_conveyor1_oneinp_new.csv')
    data_conv.loc[:, 'working'] = 1.0
    conv_emb = CachedEmbedding(LaplacianEigenmap, dim=emb_dim)
    args = {'scope': dir_with_models, 'activation': 'relu', 'layers': [64, 64], 'embedding_dim': conv_emb.dim}
    conveyor_network_ng_emb = QNetwork(graph_size, **args)
    conveyor_network_ng_emb_ws = QNetwork(graph_size, additional_inputs=[{'tag': 'working', 'dim': 1}], **args)
    conveyor_network_ng_emb_losses = qnetwork_pretrain(conveyor_network_ng_emb, shuffle(data_conv), epochs=10,
                                                       embedding=conv_emb, save_net=True)
    #conveyor_network_ng_emb_ws_losses = qnetwork_pretrain(conveyor_network_ng_emb_ws, shuffle(data_conv), epochs=20,
    #                                                      embedding=conv_emb, save_net=False)

dir_with_models = 'conveyor_test_ng'
filename_suffix = f"_{emb_dim}_{graph_size}_{os.path.split(scenario)[1]}.bin"
pretrain_filename = f"igor_pretrained{filename_suffix}"
pretrain_path = Path(TORCH_MODELS_DIR) / dir_with_models / pretrain_filename
if args.force_pretrain or not pretrain_path.exists():
    print(f"Pretraining {pretrain_path}...")
    pretrain(args, dir_with_models, pretrain_filename)
else:
    print(f"Using the already pretrained model {pretrain_path}...")


# 3. train

def run_single(file: str, router_type: str, random_seed: int, **kwargs):
    job_id = mk_job_id(router_type, random_seed)
    with tqdm(desc=job_id) as bar:
        queue = DummyProgressbarQueue(bar)
        runner = ConveyorsRunner(run_params=file, router_type=router_type, random_seed=random_seed,
                                 progress_queue=queue, **kwargs)
        event_series = runner.run(**kwargs)
    return event_series, runner

def train(args, dir_with_models: str, pretrain_filename: str, train_filename: str,
          router_type: str, retrain: bool, work_with_files: bool):
    # Igor: I did not see an easy way to change the code in a clean way
    os.environ["IGOR_OVERRIDDEN_DQN_LOAD_FILENAME"] = pretrain_filename
    os.environ["IGOR_TRAIN_PROBABILITY_SMOOTHING"] = str(args.probability_smoothing)
    
    if retrain:
        if "IGOR_OMIT_TRAINING" in os.environ:
            del os.environ["IGOR_OMIT_TRAINING"]
    else:
        os.environ["IGOR_OMIT_TRAINING"] = "True"
    
    event_series, runner = run_single(file=scenario, router_type=router_type, progress_step=500,
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
    

train_filename = f"igor_trained{filename_suffix}"
train_path = Path(TORCH_MODELS_DIR) / dir_with_models / train_filename
retrain = args.force_train or not train_path.exists()
if retrain:
    print(f"Training {train_path}...")
else:
    print(f"Using the already trained model {train_path}...")
_, world = train(args, dir_with_models, pretrain_filename, train_filename, "dqn_emb", retrain, True)


# 4. load the router graph
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

if not args.skip_graphviz:
    visualize(g)
    
sa = SymbolicAnalyzer(g, args.softmax_temperature, args.probability_smoothing,
                      args.verification_lr, delta_q_max=args.verification_max_delta_q)

print(f"Running command {args.command}...")
if args.command == "deterministic_test":
    for source in g.sources:
        for sink in g.sinks:
            print(f"Testing delivery from {source} to {sink}...")
            current_node = source
            visited_nodes = set()
            sink_embedding, _, _ = g.node_to_embeddings(sink, sink)
            while True:
                if current_node in visited_nodes:
                    print("    FAIL due to cycle")
                    break
                visited_nodes.add(current_node)
                print("    in:", current_node)
                if current_node[0] == "sink":
                    print("    ", end="")
                    print("OK" if current_node == sink else "FAIL due to wrong destination")
                    break
                elif current_node[0] in ["source", "junction"]:
                    out_nodes = g.get_out_nodes(current_node)
                    assert len(out_nodes) == 1
                    current_node = out_nodes[0]
                elif current_node[0] == "diverter":
                    current_embedding, neighbors, neighbor_embeddings = g.node_to_embeddings(current_node, sink)
                    q_values = []
                    for neighbor, neighbor_embedding in zip(neighbors, neighbor_embeddings):
                        with torch.no_grad():
                            q = g.q_forward(current_embedding, sink_embedding, neighbor_embedding).item()
                        print(f"        Q({current_node} -> {neighbor} | sink = {sink}) = {q:.4f}")
                        q_values += [q]
                    best_neighbor_index = np.argmax(np.array(q_values))
                    current_node = neighbors[best_neighbor_index]
                else:
                    raise AssertionError()
elif args.command == "embedding_adversarial":
    adv = PGDAdversary(rho=1.5, steps=100, step_size=0.02, random_start=True, stop_loss=1e5, verbose=2,
                       norm="scaled_l_2", n_repeat=2, repeat_mode="min", dtype=torch.float64)
    for sink in g.sinks:
        print(f"Measuring robustness of delivery to {sink}...")
        ma = MarkovAnalyzer(g, sink, args.simple_path_cost)
        sink_embedding, _, _ = g.node_to_embeddings(sink, sink)
        embedding_size = sink_embedding.flatten().shape[0]
        # gather all embeddings that we need to compute the objective
        stored_embeddings = OrderedDict({sink: sink_embedding})
        for node_key in ma.reachable_nodes:
            stored_embeddings[node_key], _, _ = g.node_to_embeddings(node_key, sink)

        def pack_embeddings(embedding_dict: OrderedDict) -> torch.tensor:
            return torch.cat(tuple(embedding_dict.values())).flatten()

        def unpack_embeddings(embedding_vector: torch.tensor) -> OrderedDict:
            embedding_dict = OrderedDict()
            for i, (key, value) in enumerate(stored_embeddings.items()):
                embedding_dict[key] = embedding_vector[i*embedding_size:(i + 1)*embedding_size].reshape(1, embedding_size)
            return embedding_dict

        initial_vector = pack_embeddings(stored_embeddings)

        for source in ma.reachable_sources:
            print(f"  Measuring robustness of delivery from {source} to {sink}...")
            objective, lambdified_objective = ma.get_objective(source)

            def get_gradient(x: torch.tensor) -> Tuple[torch.tensor, float, str]:
                """
                :param x: parameter vector (the one expected to converge to an adversarial example)
                Returns a tuple (gradient pointing to the direction of the adversarial attack,
                                 the corresponding loss function value,
                                 auxiliary information for printing during optimization)."""
                x = Util.optimizable_clone(x.flatten())
                embedding_dict = unpack_embeddings(x)
                objective_inputs = []
                perturbed_sink_embeddings = embedding_dict[sink].repeat(2, 1)
                for diverter in ma.nontrivial_diverters:
                    perturbed_diverter_embeddings = embedding_dict[diverter].repeat(2, 1)
                    _, current_neighbors, _ = g.node_to_embeddings(diverter, sink)
                    perturbed_neighbor_embeddings = torch.cat([embedding_dict[current_neighbor]
                                                               for current_neighbor in current_neighbors])
                    q_values = g.q_forward(perturbed_diverter_embeddings, perturbed_sink_embeddings,
                                           perturbed_neighbor_embeddings).flatten()
                    objective_inputs += [Util.q_values_to_first_probability(q_values,
                                                                            args.softmax_temperature,
                                                                            args.probability_smoothing)]
                objective_value = lambdified_objective(*objective_inputs)
                #print(objective_value.detach().cpu().numpy())
                objective_value.backward()
                aux_info = ", ".join([f"{param}={value.detach().cpu().item():.4f}"
                                      for param, value in zip(ma.params, objective_inputs)])
                return x.grad, objective_value.item(), f"[{aux_info}]"
            adv.perturb(initial_vector, get_gradient)
elif args.command == "q_adversarial":
    plot_index = 0
    for sink in g.sinks:
        print(f"Measuring robustness of delivery to {sink}...")
        ma = MarkovAnalyzer(g, sink, args.simple_path_cost)
        sink_embedding, _, _ = g.node_to_embeddings(sink, sink)
        embedding_size = sink_embedding.flatten().shape[0]
        sink_embeddings = sink_embedding.repeat(2, 1)

        for source in ma.reachable_sources:
            print(f"  Measuring robustness of delivery from {source} to {sink}...")
            objective, lambdified_objective = ma.get_objective(source)

            for node_key in g.node_keys:
                current_embedding, neighbors, neighbor_embeddings = g.node_to_embeddings(node_key, sink)

                for neighbor_key, neighbor_embedding in zip(neighbors, neighbor_embeddings):
                    # compute
                    # we assume a linear change of parameters
                    reference_q = sa.compute_gradients(current_embedding, sink_embedding,
                                                       neighbor_embedding).flatten().item()
                    actual_qs = reference_q + np.linspace(-sa.delta_q_max, sa.delta_q_max, 351)
                    kappa = sa.get_transformed_cost(ma, objective, args.cost_bound)
                    lambdified_kappa = sympy.lambdify(ma.params, kappa)
                    objective_values = torch.empty(len(actual_qs))
                    kappa_values = torch.empty(len(actual_qs))
                    for i, actual_q in enumerate(actual_qs):
                        ps = sa.compute_ps(ma, diverter, sink, sink_embeddings, reference_q, actual_q)
                        objective_values[i] = lambdified_objective(*ps)
                        kappa_values[i] = lambdified_kappa(*ps)

                    # plot
                    fig, axes = plt.subplots(2, 1, figsize=(13, 6))
                    plt.subplots_adjust(hspace=0.3)
                    caption_starts = ("Delivery cost (τ)", "Transformed delivery cost (κ)")
                    axes[0].set_yscale("log")
                    for ax, caption_start, values in zip(axes, caption_starts, (objective_values, kappa_values)):
                        label = f"{caption_start} from {source} to {sink} when making optimization step with current={node_key}, neighbor={neighbor_key}"
                        print(f"{label}...")
                        ax.set_title(label)
                        ax.plot(actual_qs, values)
                        gap = values.max() - values.min()
                        y_delta = 0 if gap > 0 else 5
                        # show the zero step value:
                        ax.vlines(reference_q, min(values) - y_delta, max(values) + y_delta)
                        ax.hlines(values[len(values) // 2], min(actual_qs), max(actual_qs))
                    # show the verification bound:
                    axes[0].hlines(args.cost_bound, min(actual_qs), max(actual_qs))
                    axes[1].hlines(0, min(actual_qs), max(actual_qs))
                    plt.savefig(f"../img/{filename_suffix}_{plot_index}.pdf")
                    plt.close()
                    print(f"Empirically found maximum of τ: {objective_values.max():.6f}")
                    print(f"Empirically found maximum of κ: {kappa_values.max():.6f}")
                    plot_index += 1
elif args.command == "q_adversarial_lipschitz":
    print(sa.net)
    sa.load_matrices()
    
    for sink in g.sinks:
        print(f"Measuring robustness of delivery to {sink}...")
        ma = MarkovAnalyzer(g, sink, args.simple_path_cost)
        sink_embedding, _, _ = g.node_to_embeddings(sink, sink)
        embedding_size = sink_embedding.flatten().shape[0]
        sink_embeddings = sink_embedding.repeat(2, 1)
        
        ps_function_names = [f"p{i}" for i in range(len(ma.params))]
        function_ps = [sympy.Function(name) for name in ps_function_names]
        evaluated_function_ps = [f(sa.beta) for f in function_ps]
        
        # cached values
        computed_logits_and_derivatives: Dict[AgentId, Tuple[sympy.Expr, sympy.Expr]] = {}
        
        def compute_logit_and_derivative(sa: SymbolicAnalyzer, diverter_key: AgentId) -> Tuple[sympy.Expr, sympy.Expr]:
            if diverter_key not in computed_logits_and_derivatives:
                diverter_embedding, _, neighbor_embeddings = g.node_to_embeddings(diverter_key, sink)
                delta_e = [sa.tensor_to_sympy(torch.cat((sink_embedding - diverter_embedding,
                                                         neighbor_embeddings[i] - diverter_embedding), dim=1).T) for i in range(2)]
                logit = sa.to_scalar(sa.sympy_q(delta_e[0]) - sa.sympy_q(delta_e[1])) / args.softmax_temperature
                dlogit_dbeta = logit.diff(sa.beta)
                computed_logits_and_derivatives[diverter_key] = logit, dlogit_dbeta
            else:
                print("      (using cached value)")
            return computed_logits_and_derivatives[diverter_key]
        
        for source in ma.reachable_sources:
            print(f"  Measuring robustness of delivery from {source} to {sink}...")
            objective, lambdified_objective = ma.get_objective(source)
        
            for node_key in g.node_keys:
                current_embedding, neighbors, neighbor_embeddings = g.node_to_embeddings(node_key, sink)

                for neighbor_key, neighbor_embedding in zip(neighbors, neighbor_embeddings):
                    print(f"    Considering learning step {node_key} -> {neighbor_key}...")
                    reference_q = sa.compute_gradients(current_embedding, sink_embedding,
                                                       neighbor_embedding).flatten().item()
                    print(f"      Reference Q value = {reference_q:.4f}")
                    sa.load_grad_matrices()
                    
                    MOCK = False
                    if MOCK:
                        dim = 7
                        #print(A.shape, b.shape, C.shape, d.shape, E.shape, f.shape)
                        sa.A = sa.A[:dim, :];    sa.A_hat = sa.A_hat[:dim, :]
                        sa.b = sa.b[:dim, :];    sa.b_hat = sa.b_hat[:dim, :]
                        sa.C = sa.C[:dim, :dim]; sa.C_hat = sa.C_hat[:dim, :dim]
                        sa.d = sa.b[:dim, :];    sa.d_hat = sa.d_hat[:dim, :]
                        sa.E = sa.E[:,    :dim]; sa.E_hat = sa.E_hat[:,    :dim]

                    print(f"      τ(p) = {objective}, τ(p) < {args.cost_bound}?")
                    kappa_of_p = sa.get_transformed_cost(ma, objective, args.cost_bound)
                    lambdified_kappa = sympy.lambdify(ma.params, kappa_of_p)
                    print(f"      κ(p) = {kappa_of_p}, κ(p) < 0?")
                    kappa_of_beta = kappa_of_p.subs(list(zip(ma.params, evaluated_function_ps)))
                    print(f"      κ(β) = {kappa_of_beta}, κ(β) < 0?")
                    dkappa_dbeta = kappa_of_beta.diff(sa.beta)
                    print(f"      dκ(β)/dβ = {dkappa_dbeta}")
                    
                    #  compute a pool of bounds
                    derivative_bounds = {}
                    for param, diverter_key in zip(ma.params, ma.nontrivial_diverters):
                        _, current_neighbors, _ = g.node_to_embeddings(diverter_key, sink)
                        print(f"      Computing the logit and its derivative for {param} = P({diverter_key} -> {current_neighbors[0]} | sink = {sink})....")
                        logit, dlogit_dbeta = compute_logit_and_derivative(sa, diverter_key)
                        
                        # surprisingly, the strings are very slow to obtain
                        if False:
                            print(f"      logit = {sa.expr_to_string(logit)[:500]} ...")
                            print(f"      dlogit/dβ = {sa.expr_to_string(dlogit_dbeta)[:500]} ...")
                            
                        print(f"      Computing logit bounds...")
                        derivative_bounds[param.name] = sa.estimate_upper_bound(dlogit_dbeta)

                    print(f"      Computing the final upper bound on dκ(β)/dβ...")
                    top_level_bound = sa.estimate_top_level_upper_bound(dkappa_dbeta, ps_function_names, derivative_bounds)
                    print(f"      Final upper bound on the Lipschitz constant of κ(β): {top_level_bound}")
                    
                    grid_size = 3
                    old_kappa_values = None
                    
                    def intermediate(x):
                        return x[1:][range(0, len(x) - 1, 2)]
                    
                    def merge_arrays(outer: np.ndarray, inner: np.ndarray) -> np.ndarray:
                        result = np.empty(len(outer) + len(inner))
                        result[range(0, len(result), 2)] = outer
                        result[1:][range(0, len(result) - 1, 2)] = inner
                        return result
                    
                    while True:
                        print(f"      Grid size = {grid_size}...")
                        beta_values = np.linspace(-sa.beta_bound, sa.beta_bound, grid_size)
                        remaining_beta_values = beta_values if old_kappa_values is None else intermediate(beta_values)
                        actual_qs = reference_q + remaining_beta_values / sa.lr / 2
                        #print(actual_qs)
                        kappa_values = np.empty(len(actual_qs))
                        for i, actual_q in enumerate(actual_qs):
                            ps = sa.compute_ps(ma, diverter, sink, sink_embeddings, reference_q, actual_q)
                            #print(ps)
                            kappa_values[i] = lambdified_kappa(*ps)
                        # 1. try to find counterexample 
                        worst_index = kappa_values.argmax()
                        worst_value = kappa_values[worst_index]
                        if worst_value >= 0:
                            print(f"        Counterexample found: q = {actual_qs[worst_index]}, Δq = {actual_qs[worst_index] - reference_q}, β = {beta_values[worst_index]}, κ = {worst_value}")
                            break
                        # 2. try to find proof
                        if old_kappa_values is not None:
                            kappa_values = merge_arrays(old_kappa_values, kappa_values)
                        kappa_upper_bound = -np.infty
                        for beta_interval, kappa_interval in zip(sa.to_intervals(beta_values), sa.to_intervals(kappa_values)):
                            max_on_interval = (top_level_bound * (beta_interval[1] - beta_interval[0]) + sum(kappa_interval)) / 2
                            kappa_upper_bound = max(kappa_upper_bound, max_on_interval)
                        print(f"        Computed upper bound on κ(β): {kappa_upper_bound}")
                        if kappa_upper_bound < 0:
                            print("        Proof found!")
                            break
                        # 3. otherwise, increase the size of the grid
                        grid_size = (grid_size - 1) * 2 + 1
                        old_kappa_values = kappa_values
elif args.command == "compare":
    _legend_txt_replace = {
        'networks': {
            'link_state': 'Shortest paths', 'simple_q': 'Q-routing', 'pred_q': 'PQ-routing',
            'glob_dyn': 'Global-dynamic', 'dqn': 'DQN', 'dqn_oneout': 'DQN (1-out)',
            'dqn_emb': 'DQN-LE', 'centralized_simple': 'Centralized control'
        }, 'conveyors': {
            'link_state': 'Vyatkin-Black', 'simple_q': 'Q-routing', 'pred_q': 'PQ-routing',
            'glob_dyn': 'Global-dynamic', 'dqn': 'DQN', 'dqn_oneout': 'DQN (1-out)',
            'dqn_emb': 'DQN-LE', 'centralized_simple': 'BSR'
        }
    }
    _targets = {'time': 'avg', 'energy': 'sum', 'collisions': 'sum'}
    _ylabels = {
        'time': 'Mean delivery time', 'energy': 'Total energy consumption', 'collisions': 'Cargo collisions'
    }
    
    router_types = ["dqn_emb", "link_state", "simple_q"]
    series = []
    for router_type in router_types:
        s, _ = train(args, dir_with_models, pretrain_filename, train_filename, router_type, True, False)
        series += [s.getSeries(add_avg=True)]
    
    dfs = []
    for router_type, s in zip(router_types, series):
        df = s.copy()
        add_cols(df, router_type=router_type, seed=args.random_seed)
        dfs.append(df)
    dfs = pd.concat(dfs, axis=0)
    
    def print_sums(df):
        types = set(df['router_type'])
        for tp in types:
            x = df.loc[df['router_type'] == tp, 'count'].sum()
            txt = _legend_txt_replace.get(tp, tp)
            print(f'  {txt}: {x}')
    
    def plot_data(data, meaning='time', figsize=(15,5), xlim=None, ylim=None,
              xlabel='Simulation time', ylabel=None, font_size=14, title=None, save_path=None,
              draw_collisions=False, context='networks', **kwargs):
        if 'time' not in data.columns:
            datas = split_dataframe(data, preserved_cols=['router_type', 'seed'])
            for tag, df in datas:
                if tag == 'collisions' and not draw_collisions:
                    print('Number of collisions:')
                    print_sums(df)
                    continue

                xlim = kwargs.get(tag+'_xlim', xlim)
                ylim = kwargs.get(tag+'_ylim', ylim)
                save_path = kwargs.get(tag+'_save_path', save_path)
                plot_data(df, meaning=tag, figsize=figsize, xlim=xlim, ylim=ylim,
                          xlabel=xlabel, ylabel=ylabel, font_size=font_size,
                          title=title, save_path=save_path, context='conveyors')
            return 

        target = _targets[meaning]
        if ylabel is None:
            ylabel = _ylabels[meaning]

        fig = plt.figure(figsize=figsize)
        ax = sns.lineplot(x='time', y=target, hue='router_type', data=data, err_kws={'alpha': 0.1})

        handles, labels = ax.get_legend_handles_labels()
        new_labels = list(map(lambda l: _legend_txt_replace[context].get(l, l), labels[1:]))
        ax.legend(handles=handles[1:], labels=new_labels, fontsize=font_size)

        ax.tick_params(axis='both', which='both', labelsize=int(font_size*0.75))

        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
        if title is not None:
            ax.set_title(title)

        ax.set_xlabel(xlabel, fontsize=font_size)
        ax.set_ylabel(ylabel, fontsize=font_size)

        if save_path is not None:
            fig.savefig('../img/' + save_path, bbox_inches='tight')
    
    plot_data(dfs, figsize=(10, 8), font_size=22, energy_ylim=(7e6, 2.3e7),
              time_save_path='conveyors-break-1-time.pdf', energy_save_path='conveyors-break-1-energy.pdf')

else:
    raise RuntimeError(f"Unknown command {args.command}.")