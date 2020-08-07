import numpy as np
import matplotlib.pyplot as plt
import argparse
import yaml
from pathlib import Path
from tqdm import tqdm
from typing import *

import torch
import pygraphviz as pgv

import os
current_dir = os.getcwd()
os.chdir("../src")
from dqnroute import *
from dqnroute.networks import *
os.chdir(current_dir)

from router_graph import RouterGraph

parser = argparse.ArgumentParser(description="Verifier of baggage routing neural networks.")
parser.add_argument("--command", type=str, required=True,
                    help="one of deterministic_test, embedding_adversarial, q_adversarial, compare")
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

parser.add_argument("--pretrain_num_episodes", type=int, default=10000,
                    help="pretrain_num_episodes (default: 10000)")

args = parser.parse_args()


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

def train(args, dir_with_models: str, pretrain_filename: str, train_filename: str, retrain: bool):
    """ ALMOST COPIED FROM THE TRAINING NOTEBOOK """
    
    def run_single(file: str, router_type: str, random_seed: int, **kwargs):
        job_id = mk_job_id(router_type, random_seed)
        with tqdm(desc=job_id) as bar:
            queue = DummyProgressbarQueue(bar)
            runner = ConveyorsRunner(run_params=file, router_type=router_type,
                                     random_seed=random_seed, progress_queue=queue, **kwargs)
            event_series = runner.run(**kwargs)
        return event_series, runner

    router_type='dqn_emb'
    #router_type='link_state'
    
    # Igor: I did not see an easy way to change the code in a clean way
    os.environ["IGOR_OVERRIDDEN_DQN_LOAD_FILENAME"] = pretrain_filename
    os.environ["IGOR_TRAIN_PROBABILITY_SMOOTHING"] = str(args.probability_smoothing)
    
    def get_net(world):
        return next(iter(next(iter(world.handlers.values())).routers.values())).brain
    
    if retrain:
        event_series, runner = run_single(file=scenario, router_type=router_type, progress_step=500,
                                          ignore_saved=[True], random_seed=args.random_seed)
    else:
        runner = ConveyorsRunner(router_type=router_type, params_override={}, run_params=scenario)
        
    world = runner.world
    net = next(iter(next(iter(world.handlers.values())).routers.values())).brain
    net._label = train_filename
    
     # Igor: save or load the trained network
    if retrain:
        net.save()
    else:
        net.restore()
    
    return world
    

train_filename = f"igor_trained{filename_suffix}"
train_path = Path(TORCH_MODELS_DIR) / dir_with_models / train_filename
retrain = args.force_train or not train_path.exists()
if retrain:
    print(f"Training {train_path}...")
else:
    print(f"Using the already trained model {train_path}...")
world = train(args, dir_with_models, pretrain_filename, train_filename, retrain)


# 4. load the router graph
g = RouterGraph(world)
print("Reachability matrix:")
g.print_reachability_matrix()

def visualize(g: RouterGraph):
    gv_graph = pgv.AGraph(directed=True)

    def get_gv_node_name(node_key: AgentId):
        return f"{node_key[0]}\n{node_key[1]}"

    for i, node_key in g.indices_to_node_keys.items():
        gv_graph.add_node(i)
        n = gv_graph.get_node(i)
        n.attr["label"] = get_gv_node_name(node_key)
        n.attr["shape"] = "box"
        n.attr["style"] = "filled"
        n.attr["fixedsize"] = "true"
        if node_key[0] == "source":
            n.attr["fillcolor"] = "#8888FF"
            n.attr["width"] = "0.6"
        elif node_key[0] == "sink":
            n.attr["fillcolor"] = "#88FF88"
            n.attr["width"] = "0.5"
        elif node_key[0] == "diverter":
            n.attr["fillcolor"] = "#FF9999"
            n.attr["width"] = "0.7"
        else:
            n.attr["fillcolor"] = "#EEEEEE"
            n.attr["width"] = "0.7"

    for from_node in g.node_keys:
        i1 = g.node_keys_to_indices[from_node]
        for to_node in g.get_out_nodes(from_node):
            i2 = g.node_keys_to_indices[to_node]
            gv_graph.add_edge(i1, i2)
            e = gv_graph.get_edge(i1, i2)
            e.attr["label"] = g.get_edge_length(from_node, to_node)

    prefix = f"../img/topology_graph{filename_suffix}."
    gv_graph.write(prefix + "gv")
    for prog in ["dot", "circo", "twopi"]:
        prog_prefix = f"{prefix}{prog}."
        for fmt in ["pdf", "png"]:
            path = f"{prog_prefix}{fmt}"
            print(f"Drawing {path} ...")
            gv_graph.draw(path, prog=prog, args="-Gdpi=300 -Gmargin=0 -Grankdir=LR")

visualize(g)

# FIXME compute node embeddings when loading the model

# TODO refactoring: extract graph drawing to somewhere

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
                        print(f"        Q({current_node} -> {neighbor} | {sink}) = {q:.4f}")
                        q_values += [q]
                    best_neighbor_index = np.argmax(np.array(q_values))
                    current_node = neighbors[best_neighbor_index]
                else:
                    raise AssertionError()
elif args.command == "embedding_adversarial":
    # TODO use the same smoothing in adversarial search
    pass
elif args.command == "q_adversarial":
    # TODO use the same smoothing in adversarial search
    pass
elif args.command == "compare":
    # TODO compare the trained model with Vyatkin/black
    pass
else:
    raise RuntimeError(f"Unknown command {args.command}.")