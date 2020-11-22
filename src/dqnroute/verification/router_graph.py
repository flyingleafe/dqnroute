from typing import *

import pygraphviz as pgv
import numpy as np
import torch

from ..utils import AgentId
from ..simulation.conveyors import ConveyorsEnvironment

from .ml_util import Util

class RouterGraph:
    """
    Clear graph representation of the conveyor network.
    """
    
    def __init__(self, world: ConveyorsEnvironment):
        # 1. explore
        self.world = world
        self.graph = world.topology_graph
        self.routers = world.handlers
        
        self.check_embeddings()
        
        # add source/diverter/sink -> router mapping
        self.node_to_router = {}
        for node_key, router_keeper in self.routers.items():
            for router_key, router in router_keeper.routers.items():
                self.node_to_router[node_key] = router_key
                self.q_network = router.brain
                self.node_repr = router._nodeRepr
                nw = router.network
        self.q_network.ff_net = Util.conditional_to_cuda(self.q_network.ff_net)
        
        # add junction -> router mapping
        for conveyor_index, m in world.conveyor_models.items():
            router_keeper = self.routers[("conveyor", conveyor_index)].routers
            router_keys = router_keeper.keys()
            junction_keys = [cp_index for cp_index, cp in m.checkpoints if cp_index[0] == "junction"]
            for router_key, junction_key in zip(router_keys, junction_keys):
                self.node_to_router[junction_key] = router_key
        
        #self.router_to_node = {v: k for k, v in self.node_to_router.items()}
        #print(sorted([(self.router_to_node[from_node], self.router_to_node[to_node]) for from_node, to_node in nw.edges()]))
        
        # increase analysis precision:
        self.q_network = self.q_network.double()
        
        # 2. load nodes and sort them
        self.node_keys: List[AgentId] = list(self.graph.nodes)
        self.node_keys = [(0, key) for key in self.node_keys if key[0] == "source"] \
                       + [(2, key) for key in self.node_keys if key[0] == "sink"] \
                       + [(1, key) for key in self.node_keys if key[0] not in ["source", "sink"]]
        self.node_keys = [x[1] for x in sorted(self.node_keys)]
        print(f"Graph size = {len(self.node_keys)}")
        self.node_keys_to_indices = {key: i for i, key in enumerate(self.node_keys)}
        self.indices_to_node_keys = {i: key for i, key in enumerate(self.node_keys)}
        
        # 3. list of nodes of particular types
        nodes_of_type = lambda s: [node_key for node_key in self.node_keys if node_key[0] == s]
        self.sources = nodes_of_type("source")
        self.sinks = nodes_of_type("sink")
        self.diverters = nodes_of_type("diverter")
        
        # 4. find edge lengths from junctions and diverter-routers 
        self.conveyor_models: dict = world.conveyor_models
        self._agent_id_to_edge_lengths = {}
        self._node_to_conveyor_ids = {node_key: set() for node_key in self.node_keys}
        for source_key in self.sources:
            self._node_to_conveyor_ids[source_key].add(world.layout["sources"][source_key[1]]["upstream_conv"])
        
        self.junctions = set()
        
        for conveyor_index, m in self.conveyor_models.items():
            checkpoints = m.checkpoints
            for cp_index, cp in enumerate(checkpoints):
                self._node_to_conveyor_ids[cp[0]].add(conveyor_index)
            
            # attribute a sink to this conveyor, if any:
            upstream = world.layout["conveyors"][conveyor_index]["upstream"]
            #print(upstream)
            if upstream["type"] == "sink":
                self._node_to_conveyor_ids[("sink", upstream["idx"])].add(conveyor_index)
            
            # add source in the beginning, if any:
            for source_index, source_dict in world.layout["sources"].items():
                if source_dict["upstream_conv"] == conveyor_index:
                    checkpoints = [(("source", source_index), 0)] + checkpoints
                    break
                    
            # add diverter in the beginning, if any:
            for diverter_index, diverter_dict in world.layout["diverters"].items():
                if diverter_dict["upstream_conv"] == conveyor_index:
                    self._node_to_conveyor_ids[("diverter", diverter_index)].add(conveyor_index)
                    checkpoints = [(("sourcing_diverter", diverter_index), 0)] + checkpoints
                    break
                    
            print(f"conveyor {conveyor_index}: {checkpoints}, length = {m.length}")
            
            for cp_index, cp in enumerate(checkpoints):
                checkpoint_node_key = cp[0]
                if checkpoint_node_key[0] == "junction":
                    self.junctions.add(checkpoint_node_key)
                position = cp[1]
                if cp_index < len(checkpoints) - 1:
                    next_position = checkpoints[cp_index + 1][1]
                else:
                    next_position = m.length
                edge_len = next_position - position
                assert edge_len > 0
                if checkpoint_node_key[0] in ["junction", "source", "diverter", "sourcing_diverter"]:
                    self._agent_id_to_edge_lengths[checkpoint_node_key] = edge_len
                    continue
                for node_key, router_keeper in self.routers.items():
                    if node_key == checkpoint_node_key:
                        routers = list(router_keeper.routers.items())
                        assert len(routers) == 1
                        router = routers[0]
                        self._agent_id_to_edge_lengths[router[0]] = edge_len
                        break
                        
        self.junctions = list(self.junctions)
        
        # add junctions also to the conveyors that they end
        for conveyor_index in self.conveyor_models.keys():
            upstream = world.layout["conveyors"][conveyor_index]["upstream"]
            if upstream["type"] == "conveyor":
                upstream_conv: int = upstream["idx"]
                upstream_pos: int = upstream["pos"]
                junctions = [junction for junction, pos in self.conveyor_models[upstream_conv].checkpoints
                             if pos == upstream_pos]
                assert len(junctions) == 1
                junction = junctions[0]
                self._node_to_conveyor_ids[junction].add(conveyor_index)
        
        # check that our node->router mapping does not violate the (reliable) networkx network
        our_router_edges = sorted([
            (self.node_to_router[from_node], self.node_to_router[to_node])
            for from_node in self.node_keys for to_node in self.get_out_nodes(from_node)
        ])
        assert our_router_edges == sorted(nw.edges()), f"{our_router_edges} != {sorted(nw.edges())}"
        #print("Edges between routers:", nw.edges())
        
        # 5. compute reachability matrix
        self.reachable = self._compute_reachability_matrix()
        
    def check_embeddings(self):
        """
        Check that the embeddings are consistent between nodes.
        """
        no_routers = sum([len(r.routers) for _, r in self.routers.items()])
        embeddings = []
        for _, router_keeper in self.routers.items():
            for _, router in router_keeper.routers.items():                
                embeddings += [np.concatenate([router.embedding.transform(i) for i in range(no_routers)])]
        
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                diff = np.abs(embeddings[i] - embeddings[j])
                if diff.max() > 0.01:
                    for embedding in embeddings:
                        print(list(embedding.round(4)))
                    print(list(diff.round(4)))
                    raise RuntimeError("Embeddings are inconsistent between nodes! This may be caused "
                                       "by the nondeterminism in computing embeddings.")
    
    def get_conveyor_of_edge(self, from_node: AgentId, to_node: AgentId) -> int:
        c_from = self._node_to_conveyor_ids[from_node]
        c_to = self._node_to_conveyor_ids[to_node]
        intersection = list(c_from.intersection(c_to))
        assert len(intersection) > 0
        assert len(intersection) == 1, \
            (f"It appears that there are two nodes {from_node} and {to_node} that are connected "
             f"by more than one conveyor section. Such situations may be possible when there is "
             f"a conveyor that starts and ends at the same other conveyor, but these situation "
             f"are not supported. A possible way to proceed is to split that diverging/converging "
             f"conveyor into two with an artificial single-input junction.")
        return intersection[0]
            
    def to_graphviz(self) -> pgv.AGraph:
        gv_graph = pgv.AGraph(directed=True)
        fill_colors = {"source": "#8888FF", "sink": "#88FF88", "diverter": "#FF9999", "junction": "#EEEEEE"}
        
        for i, node_key in self.indices_to_node_keys.items():
            gv_graph.add_node(i)
            n = gv_graph.get_node(i)
            
            r = self.node_to_router[node_key]
            label = f"{node_key[0]} {node_key[1]}\n{r[0]} {r[1]}"
            conveyor_ids = self._node_to_conveyor_ids[node_key]
            conveyor_ids = ", ".join(sorted([f"c{i}" for i in conveyor_ids]))
            label = f"{label}\n[{conveyor_ids}]"

            for k, v in {"shape": "box", "style": "filled", "fixedsize": "true", "width": "0.9",
                         "height": "0.7", "fillcolor": fill_colors[node_key[0]], "label": label}.items():
                n.attr[k] = v
        
        #for from_node in self.node_keys:
        #    print(f"{from_node} → {self.get_out_nodes(from_node)}")
        
        for from_node in self.node_keys:
            i1 = self.node_keys_to_indices[from_node]
            for to_node in self.get_out_nodes(from_node):
                i2 = self.node_keys_to_indices[to_node]
                gv_graph.add_edge(i1, i2)
                e = gv_graph.get_edge(i1, i2)
                c = self.get_conveyor_of_edge(from_node, to_node)
                e.attr["label"] = f"{self.get_edge_length(from_node, to_node)} [c{c}]"
        return gv_graph
    
    def q_forward(self, current_embedding, sink_embedding, neighbor_embedding):
        return self.q_network.forward(current_embedding, sink_embedding, neighbor_embedding)
    
    def _compute_reachability_matrix(self):
        # 1. initialize with self-reachability
        reachable = {(k1, k2): k1 == k2 for k1 in self.node_keys for k2 in self.node_keys}
        # 2. add transitions
        for from_node in self.node_keys:
            for to_node in self.get_out_nodes(from_node):
                reachable[from_node, to_node] = True
        # 3. close with Floyd-Warshall
        for k in self.node_keys:
            for i in self.node_keys:
                for j in self.node_keys:
                    reachable[i, j] |= reachable[i, k] and reachable[k, j]
        return reachable
    
    def get_edge_length(self, from_node_key: AgentId, to_node_key: AgentId) -> float:
        # Igor: the implementation of this method is not very clear.
        node_key = from_node_key
        if from_node_key[0] == "diverter":
            upstream_conv = self.world.layout["diverters"][from_node_key[1]]["upstream_conv"]
            #print(upstream_conv, self._node_to_conveyor_ids[to_node_key])
            if upstream_conv in self._node_to_conveyor_ids[to_node_key]:
                node_key = "sourcing_diverter", from_node_key[1]
        #print(from_node_key, to_node_key, "->", node_key)
        return self._agent_id_to_edge_lengths[node_key]
        
    def get_out_nodes(self, node_key: AgentId) -> List[AgentId]:
        """
        :return the list of successor nodes of node_key. If node_key is a diverter,
          then the successor nodes will be returned in the following order:
          [0] the next node along the same conveyor;
          [1] the next node along the different conveyor. 
        """
        e = sorted([e[1] for e in self.graph.out_edges(node_key)])
        if len(e) == 2:
            current_conv = self.world.layout["diverters"][node_key[1]]["conveyor"]
            if self.get_conveyor_of_edge(node_key, e[0]) != current_conv:
                e = e[::-1]
        return e
            
    def get_out_node_indices(self, node_index: int) -> List[int]:
        return [self.node_keys_to_indices[key]
                for key in self.get_out_nodes(self.indices_to_node_keys[node_index])]
    
    def print_reachability_matrix(self):
        for from_node in self.node_keys:
            for to_node in self.node_keys:
                print(1 if self.reachable[from_node, to_node] else 0, end="")
            print(f" # from {from_node}")
    
    def _get_router_embedding(self, router_key: AgentId) -> torch.Tensor:
        return Util.conditional_to_cuda(torch.DoubleTensor([self.node_repr(router_key[1])]))
    
    def node_to_embeddings(self, current_node: AgentId, sink: AgentId) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        current_router = self.node_to_router[current_node]
        current_embedding = self._get_router_embedding(current_router)
        if current_node[0] == "sink":
            out_nodes = []
        else:
            out_nodes = self.get_out_nodes(current_node)
            # leave only nodes from which the sink is reachable
            out_nodes = [out_node for out_node in out_nodes if self.reachable[out_node, sink]]
        out_embeddings = [self._get_router_embedding(self.node_to_router[out_node]) for out_node in out_nodes]
        return current_embedding, out_nodes, out_embeddings
