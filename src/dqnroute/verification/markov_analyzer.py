import sympy

from .router_graph import RouterGraph
from ..utils import AgentId

class MarkovAnalyzer:
    def __init__(self, g: RouterGraph, sink: AgentId, simple_path_cost: bool, verbose: bool = True):
        self.g = g
        # reindex nodes so that only the nodes from which the sink is reachable are considered
        self.reachable_nodes = [node_key for node_key in g.node_keys if g.reachable[node_key, sink]]
        print(f"  Nodes from which {sink} is reachable: {self.reachable_nodes}")
        self.reachable_sources = [node_key for node_key in self.reachable_nodes if node_key[0] == "source"]
        reachable_nodes_to_indices = {node_key: i for i, node_key in enumerate(self.reachable_nodes)}
        sink_index = reachable_nodes_to_indices[sink]
        #print(f"  sink index = {sink_index}")

        # filter out reachable diverters that have only one option due to shielding
        next_nodes = lambda from_key: [to_key for to_key in g.get_out_nodes(from_key) if g.reachable[to_key, sink]]
        self.nontrivial_diverters = [from_key for from_key in self.reachable_nodes if len(next_nodes(from_key)) > 1]
        assert all(node_key[0] == "diverter" for node_key in self.nontrivial_diverters)
        nontrivial_diverters_to_indices = {node_key: i for i, node_key in enumerate(self.nontrivial_diverters)}

        system_size = len(self.reachable_nodes)
        matrix = [[0 for _ in range(system_size)] for _ in range(system_size)]
        bias = [[0] for _ in range(system_size)]

        self.params = sympy.symbols([f"p{i}" for i in range(len(self.nontrivial_diverters))])
        if verbose:
            print(f"  parameters: {self.params}")

        # fill the system of linear equations
        for i in range(system_size):
            node_key = self.reachable_nodes[i]
            matrix[i][i] = 1
            if i == sink_index:
                # zero hitting time for the target sink
                assert node_key[0] == "sink"
            elif node_key[0] in ["source", "junction", "diverter"]:
                next_node_keys = next_nodes(node_key)
                if simple_path_cost:
                    bias[i][0] = 1
                if len(next_node_keys) == 1:
                    # only one possible destination
                    # either sink, junction, or a diverter with only one option due to reachability shielding
                    next_node_key = next_node_keys[0]
                    matrix[i][reachable_nodes_to_indices[next_node_key]] = -1
                    if not simple_path_cost:
                        bias[i][0] = g.get_edge_length(node_key, next_node_key)
                elif len(next_node_key) == 2:
                    # two possible destinations
                    k1, k2 = next_node_keys[0], next_node_keys[1]
                    p = self.params[nontrivial_diverters_to_indices[node_key]]
                    if verbose:
                        print(f"      {p} = P({node_key} -> {k1} | sink = {sink})" )
                        print(f"  1 - {p} = P({node_key} -> {k2} | sink = {sink})" )
                    if k1 != sink:
                        matrix[i][reachable_nodes_to_indices[k1]] = -p
                    if k2 != sink:
                        matrix[i][reachable_nodes_to_indices[k2]] = p - 1
                    if not simple_path_cost:
                        bias[i][0] = g.get_edge_length(node_key, k1) * p + g.get_edge_length(node_key, k2) * (1 - p)
                else:
                    assert False
            else:
                assert False
        matrix = sympy.Matrix(matrix)
        #print(f"  matrix: {matrix}")
        bias = sympy.Matrix(bias)
        if verbose:
            print(f"  bias: {bias}")
        self.solution = matrix.inv() @ bias
        #print(f"  solution: {self.solution}")
        
    def get_objective(self, source: AgentId):
        source_index = self.g.node_keys_to_indices[source]
        symbolic_objective = sympy.simplify(self.solution[source_index])
        print(f"    Expected delivery cost from {source} = {symbolic_objective}")
        objective = sympy.lambdify(self.params, symbolic_objective)
        return symbolic_objective, objective
