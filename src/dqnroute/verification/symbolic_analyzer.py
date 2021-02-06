import numpy as np
from typing import *

import sympy
import torch

from ..utils import AgentId

from .ml_util import Util
from .router_graph import RouterGraph
from .markov_analyzer import MarkovAnalyzer


class relu(sympy.Function):
    """
    ReLU element-wise function for SymPy.    
    """
    
    @classmethod
    def eval(cls, x):
        return x.applyfunc(lambda elem: sympy.Max(elem, 0))

    def _eval_is_real(self):
        return True


class SymbolicAnalyzer:
    """
    This class performs symbolic computations with the neural networks used in DQNroute-LE.
    These computations are used to verify the robustness of the delivery to the error in
    Q value prediction. It is assumed that the neural network is learned with stochastic
    gradient descent (SGD).
    """
    
    def __init__(self, g: RouterGraph, softmax_temperature: float, probability_smoothing: float,
                 lr: float, delta_q_max: float):
        """
        Constructs SymbolicAnalyzer.
        :param g: RouterGraph
        :param softmax_temperature: temperature (T) hyperparameter.
        :param probability_smoothing: probability smoothing parameter (between 0 and 1).
        :param lr: the learning rate of SGD.
        :param delta_q_max: maximum allowed Q value discrepancy considered in the verification
            problem.
        """
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
        """
        Converts a PyTorch matrix (or a column vector) to a SymPy matrix,
        :param x: PyTorch matrix or a vector.
        :return: SymPy matrix.
        """
        return sympy.Matrix(Util.to_numpy(x))
    
    def _layer_to_sympy(self, x: torch.nn.Linear) -> Tuple[sympy.Matrix, sympy.Matrix]:
        """
        Returns the parameters of a Linear PyTorch layer as SymPy matrices.
        :param x: Linear PyTorch layer.
        :return: (weight matrix of x as a SymPy matrix, bias vector of x as a SymPy matrix)
        """
        return self.tensor_to_sympy(x.weight), self.tensor_to_sympy(x.bias)
    
    def _layer_grad_to_sympy(self, x: torch.nn.Linear) -> sympy.Matrix:
        """
        Returns the gradients of the parameters of a Linear PyTorch layer as SymPy matrices.
        :param x:  Linear PyTorch layer.
        :return: (gradient of the weight matrix of x as a SymPy matrix,
                  gradient of the bias vector of x as a SymPy matrix).
        """
        return self.tensor_to_sympy(x.weight.grad), self.tensor_to_sympy(x.bias.grad)
    
    def load_matrices(self):
        """
        Convert the parameters of the neural network to SymPy.
        """
        self.A, self.b = self._layer_to_sympy(self.net.fc1)
        self.C, self.d = self._layer_to_sympy(self.net.fc2)
        self.E, self.f = self._layer_to_sympy(self.net.output)
    
    def load_grad_matrices(self):
        """
        Convert the gradients of the parameters of the neural network to SymPy.
        """
        self.A_hat, self.b_hat = self._layer_grad_to_sympy(self.net.fc1)
        self.C_hat, self.d_hat = self._layer_grad_to_sympy(self.net.fc2)
        self.E_hat, self.f_hat = self._layer_grad_to_sympy(self.net.output)
    
    def sympy_q(self, x: sympy.Expr) -> sympy.Expr:
        """
        Apply the SymPy representation to the neural network to the SymPy representation
        of its input.
        :param x: input vector (SymPy expression).
        :return: output of the neural network (SymPy expression).
        """
        result = relu((self.A + self.beta * self.A_hat) @ x      + self.b + self.beta * self.b_hat)
        result = relu((self.C + self.beta * self.C_hat) @ result + self.d + self.beta * self.d_hat)
        return        (self.E + self.beta * self.E_hat) @ result + self.f + self.beta * self.f_hat
    
    def mock_matrices(self, dim: int):
        """
        To be used during testing. Reduce the internal shapes of matrices to the given dimension.
        This will make computations faster but also incorrect.
        :param dim: maximum width and height of internal matrices to be ensured.
        """
        self.A = self.A[:dim, :];    self.A_hat = self.A_hat[:dim, :]
        self.b = self.b[:dim, :];    self.b_hat = self.b_hat[:dim, :]
        self.C = self.C[:dim, :dim]; self.C_hat = self.C_hat[:dim, :dim]
        self.d = self.b[:dim, :];    self.d_hat = self.d_hat[:dim, :]
        self.E = self.E[:,    :dim]; self.E_hat = self.E_hat[:,    :dim]
    
    def compute_gradients(self, curent_embedding: torch.Tensor, sink_embedding: torch.Tensor,
                          neighbor_embedding: torch.Tensor) -> torch.Tensor:
        """
        Computes the Q value based on the given embeddings, and ensures that the parameters of the
        neural network have their gradients w.r.t. the result computed.
        :param current_embedding: embedding of the current node.
        :param sink_embedding: embedding of the sink.
        :param neighbor_embedding: embedding of the chosen successor of the current node.
        :return: Q value computed for the given embeddings.
        """
        opt = torch.optim.SGD(self.g.q_network.parameters(), lr=self.lr)
        opt.zero_grad()
        predicted_q = self.g.q_forward(curent_embedding, sink_embedding, neighbor_embedding).flatten()
        predicted_q.backward()
        return predicted_q
    
    @torch.no_grad()
    def _gd_step(self, predicted_q: torch.Tensor, actual_q: torch.Tensor, reverse: bool):
        """
        Imitates a single step of SGD with the loss computed as the MSE of the Q value.
        Changes the parameters of the neural network.
        :param predicted_q: Q value predicted by the neural network.
        :param actual_q: reference Q value.
        :param reverse: whether to perform gradient ascent instead of descent.
        """
        for param in self.g.q_network.parameters():
            if param.grad is not None:
                mse_gradient = 2 * (predicted_q - actual_q) * param.grad
                param -= (-1 if reverse else 1) * self.lr * mse_gradient
    
    @torch.no_grad()
    def compute_ps(self, ma: MarkovAnalyzer, sink: AgentId, sink_embeddings: torch.Tensor,
                   predicted_q: torch.Tensor, actual_q: torch.Tensor) -> List[float]:
        """
        Compute probabilities after a single step of SGD.
        :param ma: MarkovAnalyzer constructed for the chosen verification problem.
        :param sink: the sink of the chosen verification problem.
        :param sink_embeddings: the embedding of sink.
        :param predicted_q: Q value predicted by the neural network.
        :param actual_q: reference Q value.
        :return: list of computed routing probabilities.
        """
        ps = []
        self._gd_step(predicted_q, actual_q, False)
        for diverter in ma.nontrivial_diverters:
            diverter_embedding, current_neighbors, neighbor_embeddings = self.g.node_to_embeddings(diverter, sink)
            diverter_embeddings = diverter_embedding.repeat(2, 1)
            neighbor_embeddings = torch.cat(neighbor_embeddings, dim=0)
            q_values = self.g.q_forward(diverter_embeddings, sink_embeddings, neighbor_embeddings).flatten()
            ps += [Util.q_values_to_first_probability(q_values, self.softmax_temperature,
                                                      self.probability_smoothing).item()]
        self._gd_step(predicted_q, actual_q, True)
        return ps
    
    def round_expr(self, expr: sympy.Expr, num_digits: int) -> sympy.Expr:
        """
        Rounds all the numbers in a SymPy expression. This is useful for pretty-printing.
        :param expr: SymPy expression to round.
        :param num_digits: number of digits to leave.
        :return: expression with rounded numbers.
        """
        return expr.xreplace({n : round(n, num_digits) for n in expr.atoms(sympy.Number)})

    def expr_to_string(self, expr: sympy.Expr) -> str:
        """
        Convert a SymPy expression into a prettified string.
        :param expr: SymPy expression.
        :return: string version of expr.
        """
        return str(self.round_expr(expr, 2)).replace("Max(0, ", "ReLU(")
    
    def to_intervals(self, points: List[float]) -> List[Tuple[float, float]]:
        """
        Convert a sorted list of numbers to a list of pairs of adjacent numbers.
        :param points: non-empty sorted list of numbers.
        :return: list of pairs of adjacent numbers.
        """
        return list(zip(points, points[1:]))
    
    def get_subs_value(self, interval: Tuple[float, float]) -> float:
        """
        Returns the middle of an interval.
        :param interval: interval (pair of numbers)
        :return: the middle of interval.
        """
        return (interval[1] - interval[0]) / 2
    
    def _dummy_solve_expression(self, expr: sympy.Expr) -> List[float]:
        """
        This is a solver for linear 1-variable expressions of a particular kind that SymPy
        produces after simpications. The existence of this method may look stupid, but
        sympy.solve solves such expressions very slowly.
        :param expr: linear 1-variable expressions to solve.
        :return: the value x of the single parameter of expr such that expr[x] == 0,
            wrapped into a list.
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
    
    def get_bottom_decision_points(self, expr: sympy.Expr) -> Set[float]:
        """
        Find some nonlinearity points of an expression over a single parameter.
        :param expr: the expression to analyze.
        :return: set of nonlinearity points of expr, but only those that result from
            "bottom" expressions - the ones whose subexpressions are linear.
        """
        result_set = set()
        for arg in expr.args:
            result_set.update(self.get_bottom_decision_points(arg))
        if len(result_set) == 0 and type(expr) in [sympy.Heaviside, sympy.Max]:
            # new decision point
            index = 1 if type(expr) == sympy.Max else 0
            # even though the expressions are simple, this works very slowly:
            #solutions = sympy.solve(expr.args[index], self.beta)
            solutions = self._dummy_solve_expression(expr.args[index])
            result_set.add(solutions[0])
        return result_set
    
    def resolve_bottom_decisions(self, expr: sympy.Expr, beta_point: float) -> sympy.Expr:
        """
        Resolve "bottom" Heaviside/Max subexpressions of an expression - the ones whose subexpressions
        are linear.   
        :param expr: expression to be simplified.
        :param beta_point: value of the parameter for which the expression is simplified.
        :return: expr with "bottom" subexpressions simplified.
        """
        if type(expr) in [sympy.Float, sympy.Integer, sympy.numbers.NegativeOne, sympy.Symbol]:
            return expr
        if type(expr) in [sympy.Heaviside, sympy.Max]:
            return expr.subs(self.beta, beta_point).simplify()
        arg_expressions = [self.resolve_bottom_decisions(arg, beta_point) for arg in expr.args]
        #print(type(expr), arg_expressions)
        return type(expr)(*arg_expressions)
    
    def interval_to_string(self, interval: Tuple[float, float]) -> str:
        """
        Convert an interval to a string.
        :param interval: pair of numbers.
        :return: string representation of the interval.
        """
        return f"[{interval[0]:.6f}, {interval[1]:.6f}]"
    
    def interval_list_to_string(self, interval_list: List[Tuple[float, float]]) -> str:
        """
        Convert a list of intervals to a string.
        :param interval_list: list of number pairs.
        :return: string representation of the input.
        """
        return ", ".join([self.interval_to_string(interval) for interval in interval_list])
    
    def estimate_upper_bound(self, expr: sympy.Expr) -> float:
        """
        Estimate the upper bound on the absolute value of a SymPy expression over a single
        parameter by simplifying it on various intervals and solving the obtained quadratic
        functions. This works only on very specific expressions. To be used inside
        estimate_top_level_upper_bound().
        :param expr: input SymPy expression.
        :return: the upper bound on the absolute value of expr.
        """
        points = [p for p in self.get_bottom_decision_points(expr) if np.abs(p) < self.beta_bound]
        points = [-self.beta_bound] + sorted(points) + [self.beta_bound]
        intervals = self.to_intervals(points)
        print(f"    intervals: {self.interval_list_to_string(intervals)}")
        all_values = set()
        
        for interval in intervals:
            print(f"    {self.interval_to_string(interval)}")
            e = self.resolve_bottom_decisions(expr, self.get_subs_value(interval)).simplify()
            final_decision_points = self.get_bottom_decision_points(e)
            final_decision_points = [p for p in final_decision_points if interval[0] < p < interval[1]]
            final_decision_points = [interval[0]] + final_decision_points + [interval[1]]
            refined_intervals = self.to_intervals(final_decision_points)
            for refined_interval in refined_intervals:
                refined_e = self.resolve_bottom_decisions(e, self.get_subs_value(refined_interval)).simplify()
                derivative = refined_e.diff(self.beta)
                # find stationary points of the derivative
                additional_points = [p for p in sympy.solve(derivative, self.beta)
                                     if refined_interval[0] < p < refined_interval[1]]
                all_points = [refined_interval[0]] + additional_points + [refined_interval[1]]
                all_values.update([np.abs(float(refined_e.subs(self.beta, p).simplify())) for p in all_points])
                print(f"      {self.interval_to_string(refined_interval)}: κ'(β) = {self.expr_to_string(refined_e)};"
                      f" stationary points: {additional_points}")
        return max(all_values)
    
    def estimate_top_level_upper_bound(self, expr: sympy.Expr, ps_function_names: List[str],
                                       derivative_bounds: dict) -> float:
        """
        Estimate the upper bound on the absolute value of the SymPy expression over a single parameter.
        This function supports only very specific expression that are produced during neural network
        verification.
        :param expr: input SymPy expression.
        :param ps_function_names: names of functions inside expr that correspond to routing
            probabilities.
        :param derivative_bounds: precomputed upper bounds on the absolute values of the
            derivatives of functions inside expr that correspond to routing probabilities.
        :return: the upper bound on the absolute value of expr.
        """
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
    
    def to_scalar(self, x) -> object:
        """
        Retrieve the only item of a matrix.
        :param x: matrix with one element.
        :return: the unique element of x.
        """
        assert x.shape == (1, 1)
        return x[0, 0]
    
    def get_transformed_cost(self, ma: MarkovAnalyzer, cost: sympy.Expr,
                             cost_bound: float) -> Tuple[sympy.Expr, Callable]:
        """
        Transforms the cost (τ) expression to get rid of the denominator.
        :param ma: MarkovAnalyzer.
        :param cost: original expected delivery cost (τ) SymPy expression.
        :param cost_bound: bound on cost to be verified.
        :return: (transformed cost (κ) SymPy expression, its Callable version).
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
        print(f"    nominator(p) = {nominator}")
        print(f"    denominator(p) = {denominator}")
        
        # compute the sign of v, then ensure that it is "+"
        # the values to substitute are arbitrary within (0, 1)
        denominator_value = denominator.subs([(param, 0.5) for param in ma.params]).simplify()
        print(f"    denominator({', '.join(['0.5'] * len(ma.params))}) = {float(denominator_value):.4f}")
        if denominator_value < 0:
            nominator *= -1
            denominator *= -1
        kappa = nominator - cost_bound * denominator
        return kappa, sympy.lambdify(ma.params, kappa)


class LipschitzBoundComputer:
    """
    Verifies the given bound on the expected bag delivery time ("cost"), assuming that the
    neural network changes during a single step of stochastic gradient descent (SGD).
    """
    
    def __init__(self, sa: SymbolicAnalyzer, ma: MarkovAnalyzer, objective: sympy.Expr, sink: AgentId,
                 current_embedding: torch.Tensor, sink_embedding: torch.Tensor, neighbor_embedding: torch.Tensor,
                 cost_bound: float, mock_matrices: bool = False):
        """
        Constructs LipschitzBoundComputer.
        :param sa: SymbolicAnalyzer.
        :param ma: MarkovAnalyzer.
        :param objective: the SymPy expression for the expected bag delivery time ("cost").
        :param sink_embeddings: the embedding of the sink.
        :param current_embedding: embedding of the node where the learning step is performed.
        :param sink_embedding:
        :param neighbor_embedding:
        :param cost_bound: the bound on the expected bag delivery time to be verified.
        :param mock_matrices: whether to reduce the dimension of all the matrices, to be used
            to make smoke testing faster.
        """
        self.sa = sa
        self.ma = ma
        self.objective = objective
        self.sink = sink
        self.sink_embedding = sink_embedding
        
        self.sink_embeddings = sink_embedding.repeat(2, 1)
        self.computed_logits_and_derivatives: Dict[AgentId, Tuple[sympy.Expr, sympy.Expr]] = {}

        self.reference_q = sa.compute_gradients(current_embedding, sink_embedding,
                                                neighbor_embedding).flatten().item()
        print(f"    Reference Q value = {self.reference_q:.4f}")
        
        sa.load_grad_matrices()
        if mock_matrices:
            sa.mock_matrices(7)
            
        self.ps_function_names = [f"p{i}" for i in range(len(ma.params))]
        function_ps = [sympy.Function(name) for name in self.ps_function_names]
        evaluated_function_ps = [f(sa.beta) for f in function_ps]
            
        print(f"    τ(p) = {objective}, τ(p) < {cost_bound}?")
        kappa_of_p, self.lambdified_kappa = sa.get_transformed_cost(ma, objective, cost_bound)
        print(f"    κ(p) = {kappa_of_p}, κ(p) < 0?")
        kappa_of_beta = kappa_of_p.subs(list(zip(ma.params, evaluated_function_ps)))
        print(f"    κ(β) = {kappa_of_beta}, κ(β) < 0?")
        self.dkappa_dbeta = kappa_of_beta.diff(sa.beta)
        print(f"    dκ(β)/dβ = {self.dkappa_dbeta}")
        
        self.empirical_bound, self.max_depth, self.no_evaluations, self.checked_q_measure = [None] * 4
    
    def _compute_logit_and_derivative(self, diverter_key: AgentId) -> Tuple[sympy.Expr, sympy.Expr]:
        """
        Computed the logit (the value before applying softmax/sigmoid) and its derivative for the
        chosen routing probability.
        :param diverter_key: key of the diverter that corresponds to the chosen routing probability.
        :return: (SymPy expression for the logit,
                  SymPy expression for the derivative of the logit w.r.t. β).
        """
        if diverter_key not in self.computed_logits_and_derivatives:
            diverter_embedding, _, neighbor_embeddings = self.sa.g.node_to_embeddings(diverter_key, self.sink)
            delta_e = [self.sa.tensor_to_sympy(Util.transform_embeddings(
                self.sink_embedding, diverter_embedding, neighbor_embeddings[i]).T) for i in range(2)]
            logit = self.sa.to_scalar(self.sa.sympy_q(delta_e[0]) -
                                      self.sa.sympy_q(delta_e[1])) / self.sa.softmax_temperature
            dlogit_dbeta = logit.diff(self.sa.beta)
            self.computed_logits_and_derivatives[diverter_key] = logit, dlogit_dbeta
        else:
            print("    (using cached value)")
        return self.computed_logits_and_derivatives[diverter_key]
    
    def prove_bound(self, no_points_for_presearch: int) -> bool:       
        """
        Solves the verification problem specified during construction.
        :param no_points_for_presearch: before detailed analysis, the analyzed expression will be computed
            on this number of points on a uniform grid in hope that a counterexample will be found.  
        :return: verification result as Boolean.
        """
        # for recursive executions:
        self.empirical_bound = -np.infty
        self.max_depth = 0
        self.no_evaluations = 0
        self.checked_q_measure = 0.0

        left_q, right_q = np.array([-1, 1]) * self.sa.delta_q_max + self.reference_q

        # pre-search: check uniformly positioned points hoping to find a counterexample
        if no_points_for_presearch > 2:
            print(f"    Performing pre-search of counterexamples with {no_points_for_presearch} points...")
            points = np.linspace(left_q, right_q, no_points_for_presearch)
            for q in sorted(points, key=(lambda x: -np.abs(x - self.reference_q))):
                kappa = self._q_to_kappa(q)
                if kappa >= 0:
                    self._report_counterexample(q, kappa)
                    return False
            print(f"    No counterexample found during pre-search")
        
        #  compute a pool of bounds
        derivative_bounds = {}
        for param, diverter_key in zip(self.ma.params, self.ma.nontrivial_diverters):
            _, current_neighbors, _ = self.sa.g.node_to_embeddings(diverter_key, self.sink)
            print(f"    Computing the logit and its derivative for {param} ="
                  f" P({diverter_key} → {current_neighbors[0]} | sink = {self.sink})....")
            logit, dlogit_dbeta = self._compute_logit_and_derivative(diverter_key)
            # surprisingly, the strings are very slow to obtain
            #print(f"      logit = {sa.expr_to_string(logit)[:500]} ...")
            #print(f"      dlogit/dβ = {sa.expr_to_string(dlogit_dbeta)[:500]} ...")
            print(f"    Computing logit bounds...")
            derivative_bounds[param.name] = self.sa.estimate_upper_bound(dlogit_dbeta)

        print(f"    Computing the final upper bound on dκ(β)/dβ...")
        top_level_bound = self.sa.estimate_top_level_upper_bound(self.dkappa_dbeta, self.ps_function_names,
                                                                 derivative_bounds)
        print(f"    Final upper bound on the Lipschitz constant of κ(β): {top_level_bound}")
        return self._prove_bound(left_q, right_q, self._q_to_kappa(left_q), self._q_to_kappa(right_q), 0,
                                 top_level_bound)
    
    def _q_to_kappa(self, actual_q: float) -> float:
        """
        Convert Q value to the corresponding value of the transformed cost (κ).
        :param actual_q: Q value (the one used as a target during the step of SGD).
        :return: the value of the transformed cost (κ).
        """
        self.no_evaluations += 1
        ps = self.sa.compute_ps(self.ma, self.sink, self.sink_embeddings, self.reference_q, actual_q)
        return self.lambdified_kappa(*ps)
                    
    def _q_to_beta(self, actual_q: float) -> float:
        """
        Convert Q value to β, which is a zero-centered parameter.
        :param actual_q: Q value (the one used as a target during the step of SGD).
        :return: the corresponding value of β.
        """
        return (actual_q - self.reference_q) * self.sa.lr
    
    def _report_counterexample(self, q: float, kappa: float):
        """
        Print a counterexample.
        :param q: Q value within the allowed discrepancy.
        :param kappa: the found value of the transformed cost (κ).
        """
        dq = q - self.reference_q
        beta = self._q_to_beta(q)
        print(f"    Counterexample found: q = {q:.6f}, Δq = {dq:.6f}, β = {beta:.6f}, κ = {kappa:.6f}")
    
    def _prove_bound(self, left_q: float, right_q: float, left_kappa: float, right_kappa: float,
                     depth: int, top_level_bound: float) -> bool:
        """
        Solves the verification problem specified during construction on an interval of values on the
        single parameter of the verified expression (Q value).
        :param left_q: left bound on the Q values to be checked.
        :param right_q: right bound on the Q values to be checked.
        :param left_kappa: κ(left_q).
        :param right_kappa: κ(right_q).
        :param depth: current depth of recursion.
        :param top_level_bound: bound on the derivative of the analyzed expression.
        :return: verification result as Boolean.
        """
        mid_q = (left_q + right_q) / 2
        mid_kappa = self._q_to_kappa(mid_q)
        actual_qs    = np.array([left_q,     mid_q,     right_q])
        kappa_values = np.array([left_kappa, mid_kappa, right_kappa])
        worst_index = kappa_values.argmax()
        max_kappa = kappa_values[worst_index]
        # 1. try to find counterexample
        if max_kappa >= 0:
            worst_q = actual_qs[worst_index]
            self._report_counterexample(worst_q, max_kappa)
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
        self.max_depth = max(self.max_depth, depth)
        self.empirical_bound = max(self.empirical_bound, max_kappa)
        if self.no_evaluations % 1000 == 0:
            percentage = self.checked_q_measure / self.sa.delta_q_max / 2 * 100
            print(f"    Status: {self.no_evaluations} evaluations, empirical bound is"
                  f" {self.empirical_bound:.6f}, maximum depth is {self.max_depth}, checked Δq"
                  f" percentage: {percentage:.2f}")
        
        # 3. otherwise, try recursively
        calls = [(lambda: self._prove_bound(left_q, mid_q,   left_kappa, mid_kappa,   depth + 1, top_level_bound)),
                 (lambda: self._prove_bound(mid_q,  right_q, mid_kappa,  right_kappa, depth + 1, top_level_bound))]
        # to produce counterexamples faster, start from the most empirically dangerous subinterval
        if max_on_interval.argmax() == 1:
            calls = calls[::-1]
        return calls[0]() and calls[1]()
