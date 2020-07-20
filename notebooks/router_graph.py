from ml_util import Util

import os
current_dir = os.getcwd()
os.chdir("../src")
from dqnroute import *
os.chdir(current_dir)

class RouterGraph:
    def __init__(self, runner: ConveyorsRunner):
        # 1. explore
        print(type(runner.world).mro()[:-1])
        self.world = runner.world
        self.graph = runner.world.topology_graph
        self.routers = runner.world.handlers
        self.q_network = None
        for node_key, router_keeper in self.routers.items():
            print("node", node_key, type(router_keeper).__name__)
            out_nodes = self.get_out_nodes(node_key)
            for router_key, router in router_keeper.routers.items():
                print("    router", router_key, type(router).__name__)
                self.q_network = router.brain
                self.node_repr = router._nodeRepr
        self.q_network.ff_net = Util.conditional_to_cuda(self.q_network.ff_net)
        
        # 2. load nodes and sort them
        self.node_keys: List[AgentId] = list(self.graph.nodes)
        self.node_keys = [(0, key) for key in self.node_keys if key[0] == "source"] \
                       + [(2, key) for key in self.node_keys if key[0] == "sink"] \
                       + [(1, key) for key in self.node_keys if key[0] not in ["source", "sink"]]
        self.node_keys = [x[1] for x in sorted(self.node_keys)]
        self.node_keys_to_indices = {key: i for i, key in enumerate(self.node_keys)}
        self.indices_to_node_keys = {i: key for i, key in enumerate(self.node_keys)}
        
        # 3. compute reachability matrix
        self.reachable = self._compute_reachability_matrix()
        
        # 4. list of nodes of particulat types
        nodes_of_type = lambda s: [node_key for node_key in self.node_keys if node_key[0] == s]
        self.sources = nodes_of_type("source")
        self.sinks = nodes_of_type("sink")
        self.diverters = nodes_of_type("diverter")
        
        # 5. find edge lengths from junctions and diverter-routers 
        self.conveyor_models: dict = runner.world.conveyor_models
        self._agent_id_to_edge_lengths = {}
        self._node_to_conveyor_ids = {}
        for conveyor_index, m in self.conveyor_models.items():
            checkpoints = m.checkpoints
            for cp_index, cp in enumerate(checkpoints):
                self._node_to_conveyor_ids[cp[0]] = {conveyor_index}
            
            # attribute a sink to this conveyor, if any:
            upstream = runner.world.layout["conveyors"][conveyor_index]["upstream"]
            #print(upstream)
            if upstream["type"] == "sink":
                self._node_to_conveyor_ids[("sink", upstream["idx"])] = {conveyor_index}
            
            # add source in the beginning, if any:
            for source_index, source_dict in runner.world.layout["sources"].items():
                if source_dict["upstream_conv"] == conveyor_index:
                    checkpoints = [(("source", source_index), 0)] + checkpoints
                    break
                    
            # add diverter in the beginning, if any:
            for diverter_index, diverter_dict in runner.world.layout["diverters"].items():
                if diverter_dict["upstream_conv"] == conveyor_index:
                    checkpoints = [(("sourcing_diverter", diverter_index), 0)] + checkpoints
                    break
                    
            print(f"conveyor {conveyor_index}: {checkpoints}, length = {m.length}")
                    
            for cp_index, cp in enumerate(checkpoints):
                checkpoint_node_key = cp[0]
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
        
        # add junctions also to the conveyors that they end
        for conveyor_index in self.conveyor_models.keys():
            upstream = runner.world.layout["conveyors"][conveyor_index]["upstream"]
            if upstream["type"] == "conveyor":
                upstream_conv: int = upstream["idx"]
                upstream_pos: int = upstream["pos"]
                junctions = [junction for junction, pos in self.conveyor_models[upstream_conv].checkpoints
                             if pos == upstream_pos]
                assert len(junctions) == 1
                junction = junctions[0]
                if junction in self._node_to_conveyor_ids:
                    self._node_to_conveyor_ids[junction].add(conveyor_index)
                else:
                    self._node_to_conveyor_ids[junction] = {conveyor_index}
        
        print(self._agent_id_to_edge_lengths)
        print(self._node_to_conveyor_ids)
    
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
        node_key = from_node_key
        if from_node_key[0] == "diverter":
            upstream_conv = self.world.layout["diverters"][from_node_key[1]]["upstream_conv"]
            #print(upstream_conv, self._node_to_conveyor_ids[to_node_key])
            if upstream_conv in self._node_to_conveyor_ids[to_node_key]:
                node_key = "sourcing_diverter", from_node_key[1]
        #print(from_node_key, to_node_key, "->", node_key)
        return self._agent_id_to_edge_lengths[node_key]
        
    def get_out_nodes(self, node_key: AgentId) -> list:
        return sorted([e[1] for e in self.graph.out_edges(node_key)])
    
    def get_out_node_indices(self, node_index: int) -> list:
        return [self.node_keys_to_indices[key]
                for key in self.get_out_nodes(self.indices_to_node_keys[node_index])]
    
    def print_reachability_matrix(self):
        for from_node in self.node_keys:
            for to_node in self.node_keys:
                print(1 if self.reachable[from_node, to_node] else 0, end="")
            print(f" # from {from_node}")
    
    def _get_final_router(self, node_key: AgentId) -> dict:
        """Returns dict{where_to_go: router_id}."""
        # if "source", "conveyor", "junction": return find_neighbors of the only child
        # if "sink", "diverter": it contains a router to return
        if node_key[0] in ["source", "conveyor", "junction"]:
            out_nodes = self.get_out_nodes(node_key)
            assert len(out_nodes) == 1, out_nodes
            return self._get_final_router(out_nodes[0])
        elif node_key[0] in ["sink", "diverter"]:
            r = list(self.routers[node_key].routers.keys())
            assert len(r) == 1
            return r[0]
        else:
            raise AssertionError(f"Unexpected node type: {node_key[0]}")
    
    def _get_router_embedding(self, router: AgentId):
        return Util.conditional_to_cuda(torch.FloatTensor([self.node_repr(router[1])]))
    
    def node_to_embeddings(self, current_node: AgentId, sink: AgentId) -> Tuple[torch.tensor, List[torch.tensor]]:
        current_router = list(self.routers[current_node].routers.keys())
        assert len(current_router) == 1
        current_router = current_router[0]
        current_embedding = self._get_router_embedding(current_router)
        if current_node[0] == "sink":
            out_nodes = []
        else:
            out_nodes = self.get_out_nodes(current_node)
            # leave only nodes from which the sink is reachable
            out_nodes = [out_node for out_node in out_nodes if self.reachable[out_node, sink]]
        out_embeddings = [self._get_router_embedding(self._get_final_router(out_node)) for out_node in out_nodes] 
        return current_embedding, out_nodes, out_embeddings
