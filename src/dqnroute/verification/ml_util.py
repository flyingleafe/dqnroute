import os
import random

from typing import *
from abc import ABC, abstractmethod

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch


class Util(ABC):
    """
    A convenience static class for everything that does not have its own class.
    """
    
    # if False, will use CPU even if CUDA is available
    cuda_enabled = True

    @staticmethod
    def dump_model(filename: str, o):
        """
        Dumps an object to disk with pickle.
        :param filename: filename.
        :param o: the object to pickle and write.
        """
        pickle.dump(o, gzip.open(filename, "w"), pickle.HIGHEST_PROTOCOL)
        
    @staticmethod
    def optimizable_clone(x: torch.tensor) -> torch.tensor:
        """
        Clones a PyTorch tensor and makes it suitable for optimization.
        :param x: input tensor.
        :return: x with enabled gradients.
        """
        return Util.conditional_to_cuda(x.clone().detach()).requires_grad_(True)
    
    @staticmethod
    def set_param_requires_grad(m: torch.nn.Module, value: bool):
        """
        Sets requires_grad_(value) for all parameters of the module.
        :param m: PyTorch module.
        :param value: value to set.
        """
        for p in m.parameters():
            p.requires_grad_(value)
    
    @staticmethod
    def conditional_to_cuda(x: Union[torch.tensor, torch.nn.Module]) -> torch.tensor:
        """
        Returns the tensor/module on GPU if there is at least 1 GPU, otherwise just returns the tensor.
        :param x: a PyTorch tensor or module.
        :return: x on GPU if there is at least 1 GPU, otherwise just x.
        """
        return x.cuda() if (Util.cuda_enabled and torch.cuda.is_available()) else x
    
    @staticmethod
    def number_of_trainable_parameters(model: torch.nn.Module) -> int:
        """
        Number of trainable parameters in a PyTorch module, including nested modules.
        :param model: PyTorch module.
        :return: number of trainable parameters in model.
        """
        return sum([np.prod(p.size()) for p in model.parameters() if p.requires_grad])
    
    @staticmethod
    def set_random_seed(seed: int = None):
        """
        Set random seed of random, numpy and pytorch.
        :param seed seed value. If None, it is replaced with the current timestamp.
        """
        if seed is None:
            seed = int(time.time())
        else:
            assert seed >= 0
        random.seed(seed)
        np.random.seed(seed + 1)
        torch.manual_seed(seed + 2)
        torch.cuda.manual_seed_all(seed + 2)
        
    @staticmethod
    def normalize_latent(x: torch.tensor):
        """
        Divides each latent vector of a batch by its scaled Euclidean norm.
        :param x: batch of latent vectors.
        :return normalized vector.
        """
        norm_vector = (np.sqrt(x.shape[1]) / torch.norm(x, dim=1)).unsqueeze(0)
        norm_vector = norm_vector.expand(x.shape[0], norm_vector.shape[1])
        return norm_vector @ x

    @staticmethod
    def smooth(p, alpha: float):
        # smoothing to get rid of 0 and 1 probabilities that lead to saturated gradients
        return (1 - alpha) * p  + alpha / 2

    @staticmethod
    def q_values_to_first_probability(qs: torch.tensor, temperature: float, alpha: float) -> torch.tensor:
        return Util.smooth((qs / temperature).softmax(dim=0)[0], alpha)
    
    @staticmethod
    def to_numpy(x: torch.tensor) -> np.ndarray:
        return x.detach().cpu().numpy()
