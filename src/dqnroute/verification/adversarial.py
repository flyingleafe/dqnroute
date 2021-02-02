import numpy as np
import torch
from typing import *
from abc import ABC, abstractmethod

from .ml_util import Util


class Adversary(ABC):
    """
    Base class for adversaries. Adversaries can perturb vectors given the gradient pointing to the direction
    of making the prediction worse.
    """
    
    @abstractmethod
    def perturb(self, initial_vector: torch.Tensor,
                get_gradient: Callable[[torch.Tensor], Tuple[torch.Tensor, float]]) -> torch.Tensor:
        """
        Perturb the given vector. 
        :param initial_vector: initial vector. If this is the original image representation, it must be flattened
            prior to the call (more precisely, it must be of size [1, the_rest]).
        :param get_gradient: a get_gradient function. It accepts the current vector and returns a tuple
            (gradient pointing to the direction of the adversarial attack, the corresponding loss function value).
        :return: the pertured vector of the same size as initial_vector.
        """
        pass


class PGDAdversary(Adversary):
    """
    Performes Projected Gradient Descent (PGD), or, more precisely, PG ascent according to the provided gradient.
    """
    
    def __init__(self, rho: float = 0.1, steps: int = 25, step_size: float = 0.1, random_start: bool = True,
                 stop_loss: float = 0, verbose: int = 1, norm: str = "scaled_l_2",
                 n_repeat: int = 1, repeat_mode: str = None, dtype: type = torch.float32):
        """
        Constructs PGDAdversary. 
        :param rho > 0: bound on perturbation norm.
        :param steps: number of steps to perform in each run. Less steps can be done if stop_loss is reached.
        :param step_size: step size. Each step will be of magnitude rho * step_size.
        :param random_start: if True, start search in a vector with a uniformly random radius within the rho-ball.
            Otherwise, start in the center of the rho-ball.
        :param stop_loss: the search will stop when this value of the "loss" function is exceeded.
        :param verbose: 0 (silent), 1 (regular), 2 (verbose).
        :param norm: one of 'scaled_l_2' (default), 'l_2' or 'l_inf'.
        :param n_repeat: number of times to run PGD.
        :param repeat_mode: 'any' or 'min': In mode 'any', n_repeat runs are identical and any run that reaches
            stop_loss will prevent subsequent runs. In mode 'min', all runs will be performed, and if a run
            finds a smaller perturbation according to norm, it will tighten rho on the next run.
        :param dtype: dtype.
        """
        super().__init__()
        self.rho = rho
        self.steps = steps
        self.step_size = step_size
        self.random_start = random_start
        self.stop_loss = stop_loss
        self.verbose = verbose
        # checks on norms
        assert norm in ["scaled_l_2", "l_2", "l_inf"], "norm must be either 'scaled_l_2', 'l_2' or 'l_inf'"
        self.scale_norm = norm == "scaled_l_2"
        self.inf_norm = norm == "l_inf"
        # checks on repeated runs
        assert n_repeat >= 1, "n_repeat must be positive"
        assert not(n_repeat > 1 and repeat_mode is None), "if n_repeat > 1, repeat_mode must be set" 
        assert repeat_mode in [None, "any", "min"], "if repeat_mode is set, it must be either 'any' or 'min'"
        self.n_repeat = n_repeat
        self.shrinking_repeats = repeat_mode == "min"
        self.dtype = dtype
    
    def _norm(self, x: torch.Tensor) -> float:
        """
        (Possibly scaled) norm of x.
        """
        return x.norm(np.infty if self.inf_norm else 2).item() / (np.sqrt(x.numel()) if self.scale_norm else 1)
    
    def _normalize_gradient(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalizes the vector of gradients.
        In the L2 space, this is done by dividing the vector by its norm.
        In the L-inf space, this is done by taking the sign of the gradient.
        """
        if self.inf_norm:
            return x.sign()
        else:
            norm = self._norm(x)
            if norm == 0:
                return x * 0
            return x / norm
    
    def _project(self, x: torch.Tensor, rho: float) -> torch.Tensor:
        """
        Projects the vector onto the rho-ball.
        In the L2 space, this is done by scaling the vector.
        In the L-inf space, this is done by clamping all components independently.
        """
        return x.clamp(-rho, rho) if self.inf_norm else (x / self._norm(x) * rho)
    
    def perturb(self, initial_vector: torch.Tensor,
                get_gradient: Callable[[torch.Tensor], Tuple[torch.Tensor, float, object]]) -> torch.Tensor:
        best_perturbation = None
        best_perturbation_norm = np.infty
        # rho may potentially shrink with repeat_mode == "min":
        rho = self.rho
        random_start = self.random_start
        
        # Added specifically for the verification project:
        best_objective = -float("inf")
        
        for run_n in range(self.n_repeat):
            x1 = initial_vector * 1
            perturbation = x1 * 0

            if random_start:
                # random vector within the rho-ball
                if self.inf_norm:
                    # uniform
                    perturbation = (torch.rand(1, x1.numel(), dtype=self.dtype) - 0.5) * 2 * rho
                    # possibly reduce radius to encourage search of vectors with smaller norms
                    perturbation *= np.random.rand()
                else:
                    # uniform radius, random direction
                    # note that this distribution is not uniform in terms of R^n!
                    perturbation = torch.randn(1, x1.numel(), dtype=self.dtype)
                    perturbation /= self._norm(perturbation) / rho
                    perturbation *= np.random.rand()
                perturbation = Util.conditional_to_cuda(perturbation)

            if self.verbose > 0:
                print(f">> #run = {run_n}, ║x1║ = {self._norm(x1):.5f}, ρ = {rho:.5f}")

            found = False
            for i in range(self.steps):
                #assert not torch.isnan(perturbation).any()
                perturbed_vector = x1 + perturbation
                objective_gradient, objective, aux_info = get_gradient(perturbed_vector)
                if self.verbose > 0:
                    if objective > self.stop_loss or i == self.steps - 1 or i % 5 == 0 and self.verbose > 1:
                        print(f"step {i:3d}: objective = {objective:7f}, "
                                f"║Δx║ = {self._norm(perturbation):.5f}, ║x║ = {self._norm(perturbed_vector):.5f}, {aux_info}")
                if objective > self.stop_loss:
                    found = True
                    break
                # learning step
                perturbation_step = rho * self.step_size * self._normalize_gradient(objective_gradient)
                if perturbation_step.norm() == 0:
                    print(f"zero gradient, stopping")
                    break
                perturbation += perturbation_step
                # projecting on rho-ball around x1
                if self._norm(perturbation) > rho:
                    perturbation = self._project(perturbation, rho)

            # end of run
            if found:
                if self.shrinking_repeats:
                    if self._norm(perturbation) < best_perturbation_norm:
                        best_perturbation_norm = self._norm(perturbation)
                        best_perturbation = perturbation
                        rho = best_perturbation_norm
                else: # regular repeats
                    # return immediately
                    return x1 + perturbation
            if best_perturbation is None:
                best_perturbation = perturbation
            if self.shrinking_repeats and run_n == self.n_repeat - 1:
                # heuristic: the last run is always from the center
                random_start = False
                
            # Added specifically for the verification project:
            if not found and objective > best_objective:
                best_objective = objective
                best_perturbation = perturbation
                
        return x1 + best_perturbation

