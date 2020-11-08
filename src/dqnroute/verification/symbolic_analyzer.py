import numpy as np
from typing import *

import sympy
import torch

from ..utils import AgentId

from .ml_util import Util
from .router_graph import RouterGraph
from .markov_analyzer import MarkovAnalyzer


class relu(sympy.Function):
    @classmethod
    def eval(cls, x):
        return x.applyfunc(lambda elem: sympy.Max(elem, 0))

    def _eval_is_real(self):
        return True

class SymbolicAnalyzer:
    def __init__(self, g: RouterGraph, softmax_temperature: float, probability_smoothing: float,
                 lr: float, delta_q_max: float):
        self.g = g
        self.beta = sympy.Symbol("β", real=True)
        self.net = g.q_network.ff_net
        self.softmax_temperature = softmax_temperature
        self.probability_smoothing = probability_smoothing
        self.lr = lr
        self.delta_q_max = delta_q_max
        # 2 comes from differentiating a square
        self.beta_bound = 2 * lr * delta_q_max
        self.A,     self.b,     self.C,     self.d,     self.E,     self.f     = tuple([None] * 6)
        self.A_hat, self.b_hat, self.C_hat, self.d_hat, self.E_hat, self.f_hat = tuple([None] * 6)
    
    def tensor_to_sympy(self, x: torch.Tensor) -> sympy.Matrix:
        return sympy.Matrix(Util.to_numpy(x))
    
    def _layer_to_sympy(self, x: torch.Tensor) -> sympy.Matrix:
        return self.tensor_to_sympy(x.weight), self.tensor_to_sympy(x.bias)
    
    def _layer_grad_to_sympy(self, x: torch.Tensor) -> sympy.Matrix:
        return self.tensor_to_sympy(x.weight.grad), self.tensor_to_sympy(x.bias.grad)
    
    def load_matrices(self):
        self.A, self.b = self._layer_to_sympy(self.net.fc1)
        self.C, self.d = self._layer_to_sympy(self.net.fc2)
        self.E, self.f = self._layer_to_sympy(self.net.output)
    
    def load_grad_matrices(self):
        self.A_hat, self.b_hat = self._layer_grad_to_sympy(self.net.fc1)
        self.C_hat, self.d_hat = self._layer_grad_to_sympy(self.net.fc2)
        self.E_hat, self.f_hat = self._layer_grad_to_sympy(self.net.output)
    
    def sympy_q(self, x: sympy.Expr) -> sympy.Expr:
        result = relu((self.A + self.beta * self.A_hat) @ x      + self.b + self.beta * self.b_hat)
        result = relu((self.C + self.beta * self.C_hat) @ result + self.d + self.beta * self.d_hat)
        return        (self.E + self.beta * self.E_hat) @ result + self.f + self.beta * self.f_hat
    
    def mock_matrices(self, dim: int):
        """
        To be used during testing. Reduce the internal shapes of matrices to the given dimension.
        This will make computations faster but also incorrect.
        """
        self.A = self.A[:dim, :];    self.A_hat = self.A_hat[:dim, :]
        self.b = self.b[:dim, :];    self.b_hat = self.b_hat[:dim, :]
        self.C = self.C[:dim, :dim]; self.C_hat = self.C_hat[:dim, :dim]
        self.d = self.b[:dim, :];    self.d_hat = self.d_hat[:dim, :]
        self.E = self.E[:,    :dim]; self.E_hat = self.E_hat[:,    :dim]
    
    def compute_gradients(self, curent_embedding: torch.Tensor, sink_embedding: torch.Tensor,
                          neighbor_embedding: torch.Tensor) -> torch.Tensor:
        opt = torch.optim.SGD(self.g.q_network.parameters(), lr=self.lr)
        opt.zero_grad()
        predicted_q = self.g.q_forward(curent_embedding, sink_embedding, neighbor_embedding).flatten()
        predicted_q.backward()
        return predicted_q
    
    @torch.no_grad()
    def _gd_step(self, predicted_q: torch.Tensor, actual_q: torch.Tensor, reverse: bool):
        for param in self.g.q_network.parameters():
            if param.grad is not None:
                mse_gradient = 2 * (predicted_q - actual_q) * param.grad
                param -= (-1 if reverse else 1) * self.lr * mse_gradient
    
    @torch.no_grad()
    def compute_ps(self, ma: MarkovAnalyzer, sink: AgentId, sink_embeddings: torch.Tensor,
                   predicted_q: torch.Tensor, actual_q: torch.Tensor) -> List[float]:
        """
        Compute probabilities after a single step of a gradient descent.
        """
        ps = []
        self._gd_step(predicted_q, actual_q, False)
        for diverter in ma.nontrivial_diverters:
            diverter_embedding, current_neighbors, neighbor_embeddings = self.g.node_to_embeddings(diverter, sink)
            diverter_embeddings = diverter_embedding.repeat(2, 1)
            neighbor_embeddings = torch.cat(neighbor_embeddings, dim=0)
            q_values = self.g.q_forward(diverter_embeddings, sink_embeddings, neighbor_embeddings).flatten()
            ps += [Util.q_values_to_first_probability(q_values, self.softmax_temperature, self.probability_smoothing).item()]
        self._gd_step(predicted_q, actual_q, True)
        return ps
    
    def round_expr(self, expr: sympy.Expr, num_digits: int) -> sympy.Expr:
        return expr.xreplace({n : round(n, num_digits) for n in expr.atoms(sympy.Number)})

    def expr_to_string(self, expr: sympy.Expr) -> str:
        return str(self.round_expr(expr, 2)).replace("Max(0, ", "ReLU(")
    
    def to_intervals(self, points):
        return list(zip(points, points[1:]))
    
    def get_subs_value(self, interval: Tuple[float, float]) -> float:
        return (interval[1] - interval[0]) / 2
    
    def _dummy_solve_expression(self, expr: sympy.Expr) -> List[float]:
        """
        This is a solver for linear 1-variable expressions of a particular kind that sympy
        produces after simpications. The existence of this method may look stupid, but
        sympy.solve solves such expressions very slowly.
        """
        try:
            assert type(expr) == sympy.Add, type(expr)
            assert len(expr.args) == 2, len(expr.args)
            if type(expr.args[0]) == sympy.Mul:
                product = expr.args[0]
                bias = expr.args[1]
            else:
                product = expr.args[1]
                bias = expr.args[0]
            assert len(product.args) == 2, len(product.args)
            assert type(product.args[1]) == sympy.Symbol, type(product.args[1])
            coef = product.args[0]
            assert coef != 0.0
            result = [-bias / coef]
            #print(result)
            #print(sympy.solve(expr, self.beta))
            return result
        except AssertionError:
            print(f"Warning: unusual expression {expr}, using sympy.solve instead.")
            return sympy.solve(expr, self.beta)
    
    def get_bottom_decision_points(self, expr: sympy.Expr) -> Tuple[set, bool]:
        result_set = set()
        all_args_plain = True
        for arg in expr.args:
            arg_set, arg_plain = self.get_bottom_decision_points(arg)
            result_set.update(arg_set)
            all_args_plain = all_args_plain and arg_plain
        if all_args_plain and type(expr) in [sympy.Heaviside, sympy.Max]:
            # new decision point
            index = 1 if type(expr) == sympy.Max else 0
            #solutions = [0]
            #print(expr.args[index])
            # even though the expressions are simple, this works very slowly:
            #solutions = sympy.solve(expr.args[index], self.beta)
            solutions = self._dummy_solve_expression(expr.args[index])
            #assert len(solutions) == 1
            result_set.add(solutions[0])
            #print(expr, solutions[0])
            all_args_plain = False
        return result_set, all_args_plain
    
    def resolve_bottom_decisions(self, expr: sympy.Expr, beta_point: float) -> Tuple[sympy.Expr, bool]:
        if type(expr) in [sympy.Float, sympy.Integer, sympy.numbers.NegativeOne, sympy.Symbol]:
            return expr, True
        if type(expr) in [sympy.Heaviside, sympy.Max]:
            return expr.subs(self.beta, beta_point).simplify(), False
        all_args_plain = True
        arg_expressions = []
        for arg in expr.args:
            arg_expr, arg_plain = self.resolve_bottom_decisions(arg, beta_point)
            arg_expressions += [arg_expr]
            all_args_plain = all_args_plain and arg_plain
        #print(type(expr), arg_expressions)
        return type(expr)(*arg_expressions), all_args_plain
    
    def interval_to_string(self, interval: Tuple[float, float]) -> str:
        return f"[{interval[0]:.6f}, {interval[1]:.6f}]"
    
    def interval_list_to_string(self, interval_list: List[Tuple[float, float]]) -> str:
        return ", ".join([self.interval_to_string(interval) for interval in interval_list])
    
    def estimate_upper_bound(self, expr: sympy.Expr) -> float:
        points = [p for p in self.get_bottom_decision_points(expr)[0] if np.abs(p) < self.beta_bound]
        points = sorted(points)
        points = [-self.beta_bound] + points + [self.beta_bound]
        intervals = self.to_intervals(points)
        print(f"      intervals: {self.interval_list_to_string(intervals)}")
        all_values = set()
        
        for interval in intervals:
            print(f"      {self.interval_to_string(interval)}")
            e = self.resolve_bottom_decisions(expr, self.get_subs_value(interval))[0].simplify()
            final_decision_points = self.get_bottom_decision_points(e)[0]
            final_decision_points = [p for p in final_decision_points if interval[0] < p < interval[1]]
            final_decision_points = [interval[0]] + final_decision_points + [interval[1]]
            refined_intervals = self.to_intervals(final_decision_points)
            for refined_interval in refined_intervals:
                refined_e = self.resolve_bottom_decisions(e, self.get_subs_value(refined_interval))[0].simplify()
                derivative = refined_e.diff(self.beta)
                # find stationary points of the derivative
                additional_points = [p for p in sympy.solve(derivative, self.beta)
                                     if refined_interval[0] < p < refined_interval[1]]
                all_points = [refined_interval[0]] + additional_points + [refined_interval[1]]
                all_values.update([np.abs(float(refined_e.subs(self.beta, p).simplify())) for p in all_points])
                print(f"        {self.interval_to_string(refined_interval)}: κ'(β) = {self.expr_to_string(refined_e)}; stationary points: {additional_points}")
        return max(all_values)
    
    def estimate_top_level_upper_bound(self, expr: sympy.Expr, ps_function_names: List[str],
                                       derivative_bounds: dict) -> float:
        if type(expr) == sympy.Add:
            return sum([self.estimate_top_level_upper_bound(x, ps_function_names, derivative_bounds)
                        for x in expr.args])
        if type(expr) == sympy.Mul:
            return np.prod([self.estimate_top_level_upper_bound(x, ps_function_names, derivative_bounds)
                            for x in expr.args])
        if type(expr).__name__ in ps_function_names:
            # p(beta) = (1 - smoothing_alpha) sigmoid(logit(beta)) + smoothing_alpha/2
            return 1 - self.probability_smoothing / 2
            #return plain_bounds[type(expr).__name__]
        if type(expr) == sympy.Derivative:
            assert type(expr.args[0]).__name__ in ps_function_names
            # p(beta) = (1 - smoothing_alpha) sigmoid(logit(beta)) + smoothing_alpha/2
            # p'(beta) = (1 - smoothing_alpha) sigmoid'(logit(beta)) logit'(beta)
            # the derivative of sigmoid is at most 1/4
            return derivative_bounds[type(expr.args[0]).__name__] / 4 * (1 - self.probability_smoothing)
        if type(expr) in [sympy.Float, sympy.Integer]:
            return np.abs(float(expr))
        if type(expr) == sympy.numbers.NegativeOne:
            return 1.0
        raise RuntimeError(f"Unexpected type {type(expr)} of expression {expr}")
    
    def to_scalar(self, x):
        assert x.shape == (1, 1)
        return x[0, 0]
    
    def get_transformed_cost(self, ma: MarkovAnalyzer, cost: sympy.Expr,
                             cost_bound: float) -> Tuple[sympy.Expr, Callable]:
        """
        :param ma: MarkovAnalyzer.
        :param cost: original expected delivery cost (τ) expression.
        :param cost_bound: bound in cost to be verified.
        :return: (transformed cost (κ) expression, its lambdified version).
        """
        nominator = sympy.Integer(1)
        denominator = sympy.Integer(1)
        # walk through the product
        # if this is pow(something, -1), something is the denominator
        # the rest goes to the numerator
        for arg in cost.args if type(cost) == sympy.Mul else [cost]:
            if type(arg) == sympy.Pow:
                assert type(arg.args[1]) == sympy.numbers.NegativeOne, type(arg.args[1])
                denominator *= arg.args[0]
            else:
                nominator *= arg            
        print(f"      nominator(p) = {nominator}")
        print(f"      denominator(p) = {denominator}")
        
        # compute the sign of v, then ensure that it is "+"
        # the values to subsitute are arbitrary within (0, 1)
        denominator_value = denominator.subs([(param, 0.5) for param in ma.params]).simplify()
        print(f"      denominator(0.5) = {float(denominator_value):.4f}")
        if denominator_value < 0:
            nominator *= -1
            denominator *= -1
        kappa = nominator - cost_bound * denominator
        return kappa, sympy.lambdify(ma.params, kappa)


class LipschitzBoundComputer:
    """
    Verifies the given cost upper bound for learning step robustness.
    """
    
    def __init__(self, sa: SymbolicAnalyzer, ma: MarkovAnalyzer, objective: sympy.Expr, sink: AgentId,
                 current_embedding: torch.Tensor, sink_embedding: torch.Tensor, neighbor_embedding: torch.Tensor,
                 cost_bound: float, mock_matrices: bool = False):
        self.sa = sa
        self.ma = ma
        self.objective = objective
        self.sink = sink
        self.sink_embedding = sink_embedding
        
        self.sink_embeddings = sink_embedding.repeat(2, 1)
        self.computed_logits_and_derivatives: Dict[AgentId, Tuple[sympy.Expr, sympy.Expr]] = {}

        self.reference_q = sa.compute_gradients(current_embedding, sink_embedding,
                                                neighbor_embedding).flatten().item()
        print(f"      Reference Q value = {self.reference_q:.4f}")
        
        sa.load_grad_matrices()
        if mock_matrices:
            sa.mock_matrices(7)
            
        self.ps_function_names = [f"p{i}" for i in range(len(ma.params))]
        function_ps = [sympy.Function(name) for name in self.ps_function_names]
        evaluated_function_ps = [f(sa.beta) for f in function_ps]
            
        print(f"      τ(p) = {objective}, τ(p) < {cost_bound}?")
        kappa_of_p, self.lambdified_kappa = sa.get_transformed_cost(ma, objective, cost_bound)
        print(f"      κ(p) = {kappa_of_p}, κ(p) < 0?")
        kappa_of_beta = kappa_of_p.subs(list(zip(ma.params, evaluated_function_ps)))
        print(f"      κ(β) = {kappa_of_beta}, κ(β) < 0?")
        self.dkappa_dbeta = kappa_of_beta.diff(sa.beta)
        print(f"      dκ(β)/dβ = {self.dkappa_dbeta}")
        
        self.empirical_bound, self.max_depth, self.no_evaluations, self.checked_q_measure = [None] * 4
    
    def _compute_logit_and_derivative(self, diverter_key: AgentId) -> Tuple[sympy.Expr, sympy.Expr]:
        if diverter_key not in self.computed_logits_and_derivatives:
            diverter_embedding, _, neighbor_embeddings = self.sa.g.node_to_embeddings(diverter_key, self.sink)
            delta_e = [self.sa.tensor_to_sympy(Util.transform_embeddings(
                self.sink_embedding, diverter_embedding, neighbor_embeddings[i]).T) for i in range(2)]
            logit = self.sa.to_scalar(self.sa.sympy_q(delta_e[0]) -
                                      self.sa.sympy_q(delta_e[1])) / self.sa.softmax_temperature
            dlogit_dbeta = logit.diff(self.sa.beta)
            self.computed_logits_and_derivatives[diverter_key] = logit, dlogit_dbeta
        else:
            print("      (using cached value)")
        return self.computed_logits_and_derivatives[diverter_key]
    
    def prove_bound(self) -> bool:      
        #  compute a pool of bounds
        derivative_bounds = {}
        for param, diverter_key in zip(self.ma.params, self.ma.nontrivial_diverters):
            _, current_neighbors, _ = self.sa.g.node_to_embeddings(diverter_key, self.sink)
            print(f"      Computing the logit and its derivative for {param} ="
                  f" P({diverter_key} -> {current_neighbors[0]} | sink = {self.sink})....")
            logit, dlogit_dbeta = self._compute_logit_and_derivative(diverter_key)
            # surprisingly, the strings are very slow to obtain
            #print(f"      logit = {sa.expr_to_string(logit)[:500]} ...")
            #print(f"      dlogit/dβ = {sa.expr_to_string(dlogit_dbeta)[:500]} ...")
            print(f"      Computing logit bounds...")
            derivative_bounds[param.name] = self.sa.estimate_upper_bound(dlogit_dbeta)

        print(f"      Computing the final upper bound on dκ(β)/dβ...")
        top_level_bound = self.sa.estimate_top_level_upper_bound(self.dkappa_dbeta, self.ps_function_names,
                                                                 derivative_bounds)
        print(f"      Final upper bound on the Lipschitz constant of κ(β): {top_level_bound}")
        left_q = -self.sa.delta_q_max + self.reference_q
        right_q = self.sa.delta_q_max + self.reference_q
        
        # for recursive executions:
        self.empirical_bound = -np.infty
        self.max_depth = 0
        self.no_evaluations = 2
        self.checked_q_measure = 0.0
        return self._prove_bound(left_q, right_q, self._q_to_kappa(left_q), self._q_to_kappa(right_q), 0)
    
    def _q_to_kappa(self, actual_q: float) -> float:
        ps = self.sa.compute_ps(self.ma, self.sink, self.sink_embeddings, self.reference_q, actual_q)
        return self.lambdified_kappa(*ps)
                    
    def _q_to_beta(self, actual_q: float) -> float:
        return (actual_q - self.reference_q) * self.sa.lr
    
    def _prove_bound(self, left_q: float, right_q: float, left_kappa: float, right_kappa: float,
                     depth: int) -> bool:
        mid_q = (left_q + right_q) / 2
        mid_kappa = self._q_to_kappa(mid_q)
        actual_qs    = np.array([left_q,     mid_q,     right_q])
        kappa_values = np.array([left_kappa, mid_kappa, right_kappa])
        worst_index = kappa_values.argmax()
        max_kappa = kappa_values[worst_index]
        # 1. try to find counterexample
        if max_kappa > 0:
            worst_q = actual_qs[worst_index]
            worst_dq = worst_q - reference_q
            print(f"        Counterexample found: q = {worst_q:.6f}, Δq = {worst_dq:.6f},"
                  f" β = {self._q_to_beta(worst_q):.6f}, κ = {max_kappa:.6f}")
            return False
            
        # 2. try to find proof on [left, right]
        kappa_upper_bound = -np.infty
        max_on_interval = np.empty(2)
        for i, (q_interval, kappa_interval) in enumerate(zip(self.sa.to_intervals(actual_qs),
                                                             self.sa.to_intervals(kappa_values))):
            left_beta, right_beta = self._q_to_beta(q_interval[0]), self._q_to_beta(q_interval[1])
            max_on_interval[i] = (top_level_bound * (right_beta - left_beta) + sum(kappa_interval)) / 2
        if max_on_interval.max() < 0:
            self.checked_q_measure += right_q - left_q
            return True
        
        # logging
        self.no_evaluations += 1
        self.max_depth = max(self.max_depth, depth)
        self.empirical_bound = max(self.empirical_bound, max_kappa)
        if self.no_evaluations % 100 == 0:
            percentage = self.checked_q_measure / self.sa.delta_q_max / 2 * 100
            print(f"      Status: {self.no_evaluations} evaluations, empirical bound is"
                  f" {self.empirical_bound:.6f}, maximum depth is {self.max_depth}, checked Δq"
                  f" percentage: {percentage:.2f}")
        
        # 3. otherwise, try recursively
        calls = [(lambda: self._prove_bound(left_q, mid_q,   left_kappa, mid_kappa,   depth + 1)),
                 (lambda: self._prove_bound(mid_q,  right_q, mid_kappa,  right_kappa, depth + 1))]
        # to produce counetrexamples faster,
        # start from the most empirically dangerous subinterval
        if max_on_interval.argmax() == 1:
            calls = calls[::-1]
        return calls[0]() and calls[1]()