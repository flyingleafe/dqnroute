import subprocess
import re
import os

from typing import *
from abc import ABC

import numpy as np
import scipy
import torch

from .ml_util import Util
from .router_graph import RouterGraph
from .markov_analyzer import MarkovAnalyzer
from ..utils import AgentId

import sys
sys.path.append("../NNet")
from utils.writeNNet import writeNNet


ROUND_DIGITS = 3


class VerificationResult(ABC):
    @staticmethod
    def from_marabou(marabou_lines: List[str], input_dim: int, output_dim: int):
        xs = np.zeros(input_dim)  + np.nan
        ys = np.zeros(output_dim) + np.nan
        for line in marabou_lines:
            tokens = re.split(" *= *", line)
            if len(tokens) == 2:
                #print(tokens)
                value = float(tokens[1])
                index = int(tokens[0][1:])
                symbol = tokens[0].strip()[0]
                if symbol == "x":
                    xs[index] = value
                elif symbol == "y":
                    ys[index] = value
                else:
                    raise RuntimeError(f"Unexpected assignment {line}.")
        assert np.isnan(xs).all() == np.isnan(ys).all(), \
               f"Problems with reading a counterexample (xs={xs}, ys={ys}, lines={lines})!"
        return Verified.INSTANCE if np.isnan(xs).all() else Counterexample(xs, ys)
        

class Verified(VerificationResult):
    def __str__(self):
        return "Verified"
Verified.INSTANCE = Verified()
        

class Counterexample(VerificationResult):
    def __init__(self, xs: np.ndarray, ys: np.ndarray):
        self.xs = xs
        self.ys = ys
        
    def __str__(self):
        return (f"Counterexample{{xs = {Util.list_round(self.xs, ROUND_DIGITS)},"
                               f" ys = {Util.list_round(self.ys, ROUND_DIGITS)}}}")

def verify_conjunction(*calls: Callable[[], VerificationResult]) -> VerificationResult:
    for call in calls:
        result = call()
        if type(result) == Counterexample:
            return result
    return Verified.INSTANCE


class ProbabilityRegion:
    def __init__(self, lower_bounds: np.ndarray, upper_bounds: np.ndarray, verifier: 'NNetVerifier'):
        assert lower_bounds.shape == upper_bounds.shape
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
        self.lengths = upper_bounds - lower_bounds
        assert (self.lengths > 0).all(), (self.lower_bounds, self.upper_bounds)
        self.verifier = verifier
    
    def __str__(self):
        return "Region{" + ", ".join([f"p{i} ∈ [{round(l, ROUND_DIGITS)}, {round(u, ROUND_DIGITS)}]"
                                      for i, (l, u) in enumerate(zip(self.lower_bounds,
                                                                     self.upper_bounds))]) + "}"
    
    @staticmethod
    def get_initial(size: int, verifier: 'NNetVerifier') -> 'ProbabilityRegion':
        """
        Creates the initial probability cube.
        """
        v = np.zeros(size) + verifier.probability_smoothing / 2
        return ProbabilityRegion(v, 1 - v, verifier)
    
    def get_reachability_constraints(self) -> List[str]:
        """
        Return Marabou output constraints that express the reachability of this probability region.
        """
        # convert probabilities to q value differences
        # since the probabilities are separated from 0 and 1, conversion will not produce infinities
        constraints = []
        for i in range(self.lower_bounds.shape[0]):
            lower = self.verifier.probability_to_q_diff(self.lower_bounds[i])
            upper = self.verifier.probability_to_q_diff(self.upper_bounds[i])
            expr = f"+y{2 * i} -y{2 * i + 1}"
            constraints += [f"{expr} >= {lower}"] if lower > -np.inf else []
            constraints += [f"{expr} <= {upper}"] if upper <  np.inf else []
        assert len(constraints) > 0, (
            "Got empty constraints, but the constraints must be non-trivial, which is ensured by"
            "omitting verification for the initial, always-reachable probability region.")
        return constraints
    
    def split(self) -> Tuple['ProbabilityRegion', 'ProbabilityRegion']:
        """
        Splits the probability region into two halves using the longest dimension.
        """
        longest_dimension = np.argmax(self.lengths)
        split_value = (self.lower_bounds[longest_dimension] + self.upper_bounds[longest_dimension]) / 2 
        upper1, lower2 = np.copy(self.upper_bounds), np.copy(self.lower_bounds)
        upper1[longest_dimension] = lower2[longest_dimension] = split_value
        return (ProbabilityRegion(self.lower_bounds, upper1, self.verifier),
                ProbabilityRegion(lower2, self.upper_bounds, self.verifier))

    
class NNetVerifier:
    def __init__(self, g: RouterGraph, marabou_path: str, network_filename: str, property_filename: str,
                 probability_smoothing: float, softmax_temperature: float, emb_dim: int):
        self.g = g
        self.marabou_path = marabou_path
        self.network_filename = network_filename
        self.property_filename = property_filename
        self.probability_smoothing = probability_smoothing
        self.softmax_temperature = softmax_temperature
        self.emb_dim = emb_dim
        
        # to be created right now:
        self.net, self.net_new, self.net_large = [None] * 3
        self.A    , self.B    , self.C    , self.a    , self.b    , self.c     = [None] * 6
        self.A_new, self.B_new, self.C_new, self.a_new, self.b_new, self.c_new = [None] * 6
        self.create_small_blocks()
        
        # to be created later, separatelly for each verification call:
        self.A_large, self.B_large, self.C_large, self.a_large, self.b_large, self.c_large = [None] * 6
        self.stored_embeddings, self.emb_conversion, self.node_key_to_index = [None] * 3
        self.emb_center, self.objective, self.lambdified_objective = [None] * 3
        
    @torch.no_grad()
    def create_small_blocks(self):
        """
        During formal verification, emulate computations for two neighbors simultaneously
        All embeddings here are assumed to be shifted by the current embedding.

        Computation of the first hidden layer for nbr1:
            (A11 A12) (e_sink) + (a1)
            (A21 A22) (e_nbr1)   (a2)
        Computation of the first hidden layer for nbr2:
            (A11 A12) (e_sink) + (a1)
            (A21 A22) (e_nbr2)   (a2)
        Computing both while using single e_sink:
            (A11 A12 0) (e_sink)   (a1)
            (A21 A22 0) (e_nbr1) + (a2)
            (A11 0 A12) (e_nbr2)   (a1)
            (A21 0 A22)            (a2)

        Other matrices will just be block diagonal.
        """

        self.net = self.g.q_network.ff_net
        self.A, self.B, self.C = self.net[0].weight, self.net[2].weight, self.net[4].weight
        self.a, self.b, self.c = self.net[0].bias,   self.net[2].bias,   self.net[4].bias
        A11 = self.A[:self.A.shape[0]//2,  :self.A.shape[1]//2 ]
        A12 = self.A[:self.A.shape[0]//2,   self.A.shape[1]//2:]
        A21 = self.A[ self.A.shape[0]//2:, :self.A.shape[1]//2 ]
        A22 = self.A[ self.A.shape[0]//2:,  self.A.shape[1]//2:]
        O = A11 * 0
        self.A_new = torch.cat((torch.cat((A11, A12, O), dim=1),
                                torch.cat((A21, A22, O), dim=1),
                                torch.cat((A11, O, A12), dim=1),
                                torch.cat((A21, O, A22), dim=1)), dim=0)
        self.B_new = Util.make_block_diagonal(self.B, 2)
        self.C_new = Util.make_block_diagonal(self.C, 2)
        self.a_new = torch.cat((self.a,) * 2, dim=0)
        self.b_new = torch.cat((self.b,) * 2, dim=0)
        self.c_new = torch.cat((self.c,) * 2, dim=0)
        # merely to compute the output independently, not using the verification tool:
        self.net_new = Util.to_torch_relu_nn([self.A_new, self.B_new, self.C_new],
                                             [self.a_new, self.b_new, self.c_new])
        
    @torch.no_grad()
    def create_large_blocks(self, probability_dimension: int):
        self.A_large = Util.make_block_diagonal(self.A_new, probability_dimension)
        self.A_large = self.A_large @ self.emb_conversion
        self.B_large = Util.make_block_diagonal(self.B_new, probability_dimension)
        self.C_large = Util.make_block_diagonal(self.C_new, probability_dimension)
        self.a_large = torch.cat((self.a_new,) * probability_dimension, dim=0)
        self.b_large = torch.cat((self.b_new,) * probability_dimension, dim=0)
        self.c_large = torch.cat((self.c_new,) * probability_dimension, dim=0)
        #print(A_large.shape, B_large.shape, C_large.shape, a_large.shape, b_large.shape, c_large.shape)
        self.net_large = Util.to_torch_relu_nn([self.A_large, self.B_large, self.C_large],
                                               [self.a_large, self.b_large, self.c_large])
    
    def probability_to_q_diff(self, p: float) -> float:
        unsmoothed = Util.unsmooth(p, self.probability_smoothing)
        EPS = 1e-9
        if unsmoothed <= EPS:
            return -np.infty
        if unsmoothed >= 1 - EPS:
            return np.infty
        return scipy.special.logit(unsmoothed) * self.softmax_temperature
    
    def _get_embedding_conversion(self, sink: AgentId, ma: MarkovAnalyzer):
        """
        Creates a matrix that transforms all (unshifted) embeddings to groups
        of shifted embeddings (sink, nbr1, nbr2) for each probability.
        """
        m = len(ma.params)
        n = len(self.stored_embeddings)
        I = torch.tensor(np.identity(self.emb_dim), dtype=torch.float64)
        result = torch.zeros(self.emb_dim * 3 * m, self.emb_dim * n, dtype=torch.float64)
        
        for prob_index, diverter_key in enumerate(ma.nontrivial_diverters):
            _, neighbors, _ = self.g.node_to_embeddings(diverter_key, sink)
                
            # fill the next (emb_dim * 3) rows of the matrix
            #   fill with I for the sink and neighbors
            #   fill with -I for the current node (to subtract its embedding)
                
            # sink embedding, neighbor 1 embedding, neighbor 2 embedding:
            for k, key in enumerate([sink] + neighbors):
                Util.fill_block(result, 3 * prob_index + k, self.node_key_to_index[key], I)
                # shift the embedding by the current one:
                Util.fill_block(result, 3 * prob_index + k, self.node_key_to_index[diverter_key], -I)
        return Util.conditional_to_cuda(result)
    
    def verify_cost_delivery_bound(self, sink: AgentId, source: AgentId, ma: MarkovAnalyzer,
                                   input_eps_l_inf: float, cost_bound: float) -> VerificationResult:
        sink_embedding, _, _ = self.g.node_to_embeddings(sink, sink)
        self.objective, self.lambdified_objective = ma.get_objective(source)

        # gather all embeddings that we need to compute the objective:
        self.stored_embeddings = OrderedDict({sink: sink_embedding})
        for node_key in ma.reachable_nodes:
            self.stored_embeddings[node_key], _, _ = self.g.node_to_embeddings(node_key, sink)
        
        m = len(ma.params)
        n = len(self.stored_embeddings)
        print(f"Number of embeddings: {n}, number of the probabilities: {m}")
        
        # pack the default embeddings, the input center for robustness verification:
        self.emb_center = torch.cat(tuple(self.stored_embeddings.values())).flatten()
                    
        # we also need the indices of all nodes in this embedding storage:
        self.node_key_to_index = {key: i for i, key in enumerate(self.stored_embeddings.keys())}
        assert self.node_key_to_index[sink] == 0
        
        # STAGE 1: create a matrix that transforms all (unshifted) embeddings
        # to groups of shifted embeddings (sink, nbr1, nbr2) for each probability
        self.emb_conversion = self._get_embedding_conversion(sink, ma)
        
        # STAGE 2: create matrices for each probability from blocks,
        # then multiply this matrix by the previous one
        self.create_large_blocks(m)
        
        region = ProbabilityRegion.get_initial(len(ma.params), self)
        return self._verify_cost_delivery_bound(sink, source, ma, input_eps_l_inf, cost_bound, region, True)

    def _verify_cost_delivery_bound(self, sink: AgentId, source: AgentId, ma: MarkovAnalyzer,
                                    input_eps_l_inf: float, cost_bound: float,
                                    region: ProbabilityRegion, region_is_initial: bool) -> VerificationResult:
        print(f"Verifying {region}")
        objective, lambdified_objective = ma.get_objective(source)
        m = len(ma.params)
        n = len(self.stored_embeddings)
        
        # R is our probability hyperrectange
        
        # 1. Prove or refute ∀p ∈ R cost ≤ cost_bound with CSP/SMT solvers
        region_reachable = True
        # TODO
        
        if not region_reachable:
            return False
        
        # 2. Find out whether R is reachable (for some allowed embedding)
        # if the region is initial, it is always reachable
        region_reachable = region_is_initial or self.verify_adv_robustness(
            self.net_large,
            [self.A_large, self.B_large, self.C_large],
            [self.a_large, self.b_large, self.c_large],
            self.emb_center.flatten(), input_eps_l_inf,
            region.get_reachability_constraints(), check_or=False
        )
        if not region_reachable:
            return False
        
        r1, r2 = region.split()
        run_recursively = lambda r: self._verify_cost_delivery_bound(
            sink, source, ma, input_eps_l_inf, cost_bound, r, False)
        return verify_conjunction(run_recursively(r1), run_recursively(r2))
        
    def verify_adv_robustness(self, net: torch.nn.Module, weights: List[torch.Tensor],
                              biases: List[torch.Tensor], input_center: torch.Tensor, input_eps: float,
                              output_constraints: List[str], check_or: bool) -> VerificationResult:
        if check_or:
            calls = [lambda: self.verify_adv_robustness(net, weights, biases, input_center, input_eps, 
                                                        [constraint], False) for constraint in output_constraints]
            return verify_conjunction(*calls)
        
        # write the NN
        input_dim = weights[0].shape[1]
        output_dim = biases[-1].shape[0]
        input_mins  = list(np.zeros(input_dim) - 10e6)
        input_maxes = list(np.zeros(input_dim) + 10e6)
        means  = list(np.zeros(input_dim)) + [0.]
        ranges = list(np.zeros(input_dim)) + [1.]
        writeNNet([Util.to_numpy(x) for x in weights],
                  [Util.to_numpy(x) for x in biases],
                  input_mins, input_maxes, means, ranges, self.network_filename)
        # write the property
        with open(self.property_filename, "w") as f:
            for i in range(input_dim):
                f.write(f"x{i} >= {input_center[i] - input_eps}{os.linesep}")
                f.write(f"x{i} <= {input_center[i] + input_eps}{os.linesep}")
            for constraint in output_constraints:
                f.write(constraint + os.linesep)
    
        # call Marabou
        command = [self.marabou_path, self.network_filename, self.property_filename]
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        lines = []
        while process.poll() is None:
            line = process.stdout.readline().decode("utf-8")
            print(line, end="")
            lines += [line.strip()]
        final_lines = process.stdout.read().decode("utf-8").splitlines()
        process.stdout.close()
        for line in final_lines:
            print(line)
            lines += [line.strip()]

        # construct counterexample
        result = VerificationResult.from_marabou(lines, input_dim, output_dim)
        if type(result) == Counterexample:
            with torch.no_grad():
                actual_output = Util.to_numpy(net(Util.conditional_to_cuda(torch.tensor(result.xs))))
            assert (np.abs(result.ys - actual_output) < 0.01).all(), "Output cross-check failed!"
            print(f"  counterexample: NN({Util.list_round(result.xs, ROUND_DIGITS)}) ="
                  f" {Util.list_round(result.ys, ROUND_DIGITS)} [cross-check:"
                  f" {Util.list_round(actual_output, ROUND_DIGITS)}]")
        return result
