import sympy

from typing import *

from .router_graph import RouterGraph
from ..utils import AgentId

class MarkovAnalyzer:
    def __init__(self, g: RouterGraph, source: AgentId, sink: AgentId, simple_path_cost: bool = False,
                 verbose: bool = True):
        self.g = g
        self.source = source
        self.sink = sink
        
        # remove the nodes that are not relevant for the delivery between this source and this sink
        self.reachable_nodes = [node_key for node_key in g.node_keys
                                if g.reachable[source, node_key] and g.reachable[node_key, sink]]
        
        if verbose:
            print(f"  Nodes between {source} and {sink}: {self.reachable_nodes}")
        
        self.reachable_sources = [node_key for node_key in self.reachable_nodes if node_key[0] == "source"]
        
        # reindex nodes
        self.reachable_nodes_to_indices = {node_key: i for i, node_key in enumerate(self.reachable_nodes)}
        sink_index = self.reachable_nodes_to_indices[sink]

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
                    matrix[i][self.reachable_nodes_to_indices[next_node_key]] = -1
                    if not simple_path_cost:
                        bias[i][0] = g.get_edge_length(node_key, next_node_key)
                elif len(next_node_key) == 2:
                    # two possible destinations
                    k1, k2 = next_node_keys[0], next_node_keys[1]
                    p = self.params[nontrivial_diverters_to_indices[node_key]]
                    if verbose:
                        print(f"      {p} = P({node_key} → {k1} | sink = {sink})" )
                        print(f"  1 - {p} = P({node_key} → {k2} | sink = {sink})" )
                    if k1 != sink:
                        matrix[i][self.reachable_nodes_to_indices[k1]] = -p
                    if k2 != sink:
                        matrix[i][self.reachable_nodes_to_indices[k2]] = p - 1
                    if not simple_path_cost:
                        bias[i][0] = g.get_edge_length(node_key, k1) * p + g.get_edge_length(node_key, k2) * (1 - p)
                else:
                    assert False
            else:
                assert False
        matrix = sympy.Matrix(matrix)
        bias = sympy.Matrix(bias)
        self.solution = matrix.inv() @ bias
        if verbose:
            #print(f"  matrix: {matrix}")
            print(f"  bias: {bias}")
            #print(f"  solution: {self.solution}")
        
    def get_objective(self) -> Tuple[sympy.Expr, Callable]:
        source_index = self.reachable_nodes_to_indices[self.source]
        symbolic_objective = sympy.simplify(self.solution[source_index])
        print(f"  E(delivery cost from {self.source} to {self.sink}) = {symbolic_objective}")
        objective = sympy.lambdify(self.params, symbolic_objective)
        return symbolic_objective, objective

