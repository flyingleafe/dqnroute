import os
import subprocess
import re
#import multiprocessing

from typing import *
from abc import ABC

import numpy as np
import scipy
import torch
import z3

from .ml_util import Util
from .router_graph import RouterGraph
from .markov_analyzer import MarkovAnalyzer
from .embedding_packer import EmbeddingPacker
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
        return Verified() if np.isnan(xs).all() else Counterexample(xs, ys)
        

class Verified(VerificationResult):
    def __str__(self):
        return "Verified"
        

def marabou_float2str(x: float) -> str:
    return f"{x:.15f}"


class Counterexample(VerificationResult):
    def __init__(self, xs: np.ndarray, ys: np.ndarray):
        self.xs = xs
        self.ys = ys
        self.probabilities = None
        self.objective_value = None
    
    def add_objective_value(self, probabilities: np.ndarray, objective_value: float):
        self.probabilities = probabilities
        self.objective_value = objective_value
    
    def __str__(self):
        round_list = lambda x: Util.list_round(x, ROUND_DIGITS)
        probs = "?" if self.probabilities   is None else f"{round_list(self.probabilities)}"
        obj = "?"   if self.objective_value is None else f"{self.objective_value:.4f}"
        return (f"Counterexample{{embeddings = {round_list(self.xs)},"
                               f" Q values = {round_list(self.ys)},"
                               f" probabilities = {probs}, objective = {obj}}}")

def verify_conjunction(calls: List[Callable[[], VerificationResult]]) -> VerificationResult:
    for call in calls:
        result = call()
        if type(result) == Counterexample:
            return result
    return Verified()


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
            constraints += [f"{expr} >= {marabou_float2str(lower)}"] if lower > -np.inf else []
            constraints += [f"{expr} <= {marabou_float2str(upper)}"] if upper <  np.inf else []
        #assert len(constraints) > 0, (
        #    "Got empty constraints, but the constraints must be non-trivial, which is ensured by"
        #    "omitting verification for the initial, always-reachable probability region.")
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
        self.A, self.B, self.C = [self.net[i].weight for i in [0, 2, 4]]
        self.a, self.b, self.c = [self.net[i].bias   for i in [0, 2, 4]]
        d1, d2 = self.A.shape[0] // 2, self.A.shape[1] // 2
        A11 = self.A[:d1,  :d2 ]
        A12 = self.A[:d1,   d2:]
        A21 = self.A[ d1:, :d2 ]
        A22 = self.A[ d1:,  d2:]
        O = A11 * 0
        self.A_new = torch.cat((torch.cat((A11, A12, O), dim=1),
                                torch.cat((A21, A22, O), dim=1),
                                torch.cat((A11, O, A12), dim=1),
                                torch.cat((A21, O, A22), dim=1)), dim=0)
        self.B_new, self.C_new       = [Util.make_block_diagonal(x, 2) for x in [self.B, self.C]]
        self.a_new, self.b_new, self.c_new = [Util.repeat_tensor(x, 2) for x in [self.a, self.b, self.c]]
        # merely to compute the output independently, not using the verification tool:
        self.net_new = Util.to_torch_relu_nn([self.A_new, self.B_new, self.C_new],
                                             [self.a_new, self.b_new, self.c_new])
        
    @torch.no_grad()
    def create_large_blocks(self, probability_dimension: int):
        self.A_large = Util.make_block_diagonal(self.A_new, probability_dimension)
        self.A_large = self.A_large @ self.emb_conversion
                                  
        self.B_large, self.C_large = [Util.make_block_diagonal(x, probability_dimension)
                                      for x in [self.B_new, self.C_new]]
        self.a_large, self.b_large, self.c_large = [Util.repeat_tensor(x, probability_dimension)
                                                    for x in [self.a_new, self.b_new, self.c_new]]

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
        n = self.embedding_packer.number_of_embeddings()
        I = torch.tensor(np.identity(self.emb_dim), dtype=torch.float64)
        result = torch.zeros(self.emb_dim * 3 * m, self.emb_dim * n, dtype=torch.float64)
        
        for prob_index, diverter_key in enumerate(ma.nontrivial_diverters):
            _, neighbors, _ = self.g.node_to_embeddings(diverter_key, sink)
                
            # fill the next (emb_dim * 3) rows of the matrix
            #   fill with I for the sink and neighbors
            #   fill with -I for the current node (to subtract its embedding)
                
            # sink embedding, neighbor 1 embedding, neighbor 2 embedding:
            for k, key in enumerate([sink] + neighbors):
                Util.fill_block(result, 3 * prob_index + k,
                                self.embedding_packer.node_key_to_index(key), I)
                # shift the embedding by the current one:
                Util.fill_block(result, 3 * prob_index + k,
                                self.embedding_packer.node_key_to_index(diverter_key), -I)
        return Util.conditional_to_cuda(result)
    
    def _prove_bound_for_region(self, probability_dimension: int, cost_bound: float,
                                region: ProbabilityRegion) -> bool:
        z3.set_option(rational_to_decimal=True)
        ps = [z3.Real(f"p{i}") for i in range(probability_dimension)]
        constraints = [self.lambdified_objective(*ps) >= cost_bound]
        constraints += [ps[i] >= region.lower_bounds[i] for i in range(probability_dimension)]
        constraints += [ps[i] <= region.upper_bounds[i] for i in range(probability_dimension)]
        s = z3.Solver()
        s.add(constraints)
        result = s.check()
        return str(result) == "unsat"
    
    def verify_cost_delivery_bound(self, sink: AgentId, source: AgentId, ma: MarkovAnalyzer,
                                   input_eps_l_inf: float, cost_bound: float) -> VerificationResult:
        sink_embedding, _, _ = self.g.node_to_embeddings(sink, sink)
        self.objective, self.lambdified_objective = ma.get_objective(source)

        # gather all embeddings that we need to compute the objective:
        self.embedding_packer = EmbeddingPacker(self.g, sink, sink_embedding, ma.reachable_nodes)
        # pack the default embeddings, the input center for robustness verification:
        self.emb_center = self.embedding_packer.initial_vector()
        
        m = len(ma.params)
        n = self.embedding_packer.number_of_embeddings()
        print(f"    Number of embeddings: {n}, number of probabilities: {m}")
        
        # STAGE 1: create a matrix that transforms all (unshifted) embeddings
        # to groups of shifted embeddings (sink, nbr1, nbr2) for each probability
        self.emb_conversion = self._get_embedding_conversion(sink, ma)
        
        # STAGE 2: create matrices for each probability from blocks,
        # then multiply this matrix by the previous one
        self.create_large_blocks(m)
        
        region = ProbabilityRegion.get_initial(len(ma.params), self)
        return self._verify_cost_delivery_bound(sink, source, ma, input_eps_l_inf, cost_bound, region, 0)
    
    def _verify_cost_delivery_bound(self, sink: AgentId, source: AgentId, ma: MarkovAnalyzer,
                                    input_eps_l_inf: float, cost_bound: float,
                                    region: ProbabilityRegion, depth: int) -> VerificationResult:
        print(f"    [depth={depth}] Currently verifying: {region}.")
        m = len(ma.params)
        n = self.embedding_packer.number_of_embeddings()
        
        # R is our probability hyperrectange
        
        # 1. Prove or refute ∀p ∈ R cost ≤ cost_bound with CSP/SMT solvers
        print(f"    [depth={depth}] Calling Z3...")
        if self._prove_bound_for_region(m, cost_bound, region):
            return Verified()
        
        # 2. Find out whether R is reachable (for some allowed embedding)
        print(f"    [depth={depth}] Calling Marabou...")
        result = self.verify_adv_robustness(
            self.net_large,
            [self.A_large, self.B_large, self.C_large],
            [self.a_large, self.b_large, self.c_large],
            self.emb_center.flatten(), input_eps_l_inf,
            region.get_reachability_constraints(), check_or=False
        )
        if type(result) == Verified:
            print(f"    [depth={depth}] SMT verification proved that the bound cannot be exceeeded in"
                  " this probability region.")
            return result
        assert type(result) == Counterexample
        
        # 3. If R is reachable, check whether the bound is exceeded for the found example
        # TODO run the network on the found point
        with torch.no_grad():
            xs = Util.conditional_to_cuda(torch.DoubleTensor(result.xs))
            ys = Util.conditional_to_cuda(torch.DoubleTensor(result.ys))
            embedding_dict = self.embedding_packer.unpack(xs)
            objective_value, ps = self.embedding_packer.compute_objective(
                embedding_dict, ma.nontrivial_diverters, self.lambdified_objective,
                self.softmax_temperature, self.probability_smoothing)
            objective_value = objective_value.item()
            executed_ps = [p.item() for p in ps]
            counterexample_ps = [Util.q_values_to_first_probability(ys[(2 * i):(2 * i+2)], self.softmax_temperature,
                                                                    self.probability_smoothing).item()
                                 for i in range(m)]
        print(f"    [depth={depth}] Checking candidate counterexample with"
              f" ys={Util.list_round(counterexample_ps, ROUND_DIGITS)}"
              f" [cross-check: {Util.list_round(executed_ps, ROUND_DIGITS)}]...")
        if objective_value >= cost_bound:
            result.add_objective_value(np.array(counterexample_ps), objective_value)
            print(f"    [depth={depth}] True counterexample found!")
            return result
        
        # 4. If no conclusion can be made, split R and try recursively
        print(f"    [depth={depth}] No conclusion, trying recursively...")
        r1, r2 = region.split()
        calls = [lambda: self._verify_cost_delivery_bound(
            sink, source, ma, input_eps_l_inf, cost_bound, r, depth + 1) for r in [r1, r2]]
        return verify_conjunction(calls)
        
    def verify_adv_robustness(self, net: torch.nn.Module, weights: List[torch.Tensor],
                              biases: List[torch.Tensor], input_center: torch.Tensor, input_eps: float,
                              output_constraints: List[str], check_or: bool) -> VerificationResult:
        if check_or:
            calls = [lambda: self.verify_adv_robustness(net, weights, biases, input_center, input_eps, 
                                                        [constraint], False) for constraint in output_constraints]
            return verify_conjunction(calls)
        
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
                # the format is essential, Marabou does not support the exponential format
                f.write(f"x{i} >= {marabou_float2str(input_center[i] - input_eps)}{os.linesep}")
                f.write(f"x{i} <= {marabou_float2str(input_center[i] + input_eps)}{os.linesep}")
            for constraint in output_constraints:
                f.write(constraint + os.linesep)
    
        # call Marabou
        #dnc = ["--dnc", "--num-workers", "2"]#str(multiprocessing.cpu_count())]
        dnc = []
        command = [self.marabou_path, "--verbosity", "0", *dnc, self.network_filename, self.property_filename]
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        lines = []
        while process.poll() is None:
            line = process.stdout.readline().decode("utf-8")
            #print(line, end="")
            lines += [line.strip()]
        final_lines = process.stdout.read().decode("utf-8").splitlines()
        process.stdout.close()
        for line in final_lines:
            #print(line)
            lines += [line.strip()]
        print("  ".join(lines))
            
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
