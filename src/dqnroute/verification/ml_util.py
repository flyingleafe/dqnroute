import os
import random

from typing import *
from abc import ABC, abstractmethod

import numpy as np
import torch


class Util(ABC):
    """
    A convenience static class for everything that does not have its own class.
    """
    
    # if False, will use CPU even if CUDA is available
    cuda_enabled = True

    @staticmethod
    def dump_model(filename: str, o: object):
        """
        Dumps an object to disk with pickle.
        :param filename: filename.
        :param o: the object to pickle and write.
        """
        pickle.dump(o, gzip.open(filename, "w"), pickle.HIGHEST_PROTOCOL)
        
    @staticmethod
    def optimizable_clone(x: torch.Tensor) -> torch.Tensor:
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
    def conditional_to_cuda(x: Union[torch.Tensor, torch.nn.Module]) -> torch.Tensor:
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
    def normalize_latent(x: torch.Tensor) -> torch.Tensor:
        """
        Divides each latent vector of a batch by its scaled Euclidean norm.
        :param x: batch of latent vectors.
        :return normalized vector.
        """
        norm_vector = (np.sqrt(x.shape[1]) / torch.norm(x, dim=1)).unsqueeze(0)
        norm_vector = norm_vector.expand(x.shape[0], norm_vector.shape[1])
        return norm_vector @ x
    
    @staticmethod
    def to_numpy(x: torch.Tensor) -> np.ndarray:
        return x.detach().cpu().numpy()

    @staticmethod
    def to_torch_linear(weight: torch.Tensor, bias: torch.Tensor) -> torch.nn.Linear:
        m = torch.nn.Linear(*weight.shape)
        m.weight = torch.nn.Parameter(weight)
        m.bias   = torch.nn.Parameter(bias)
        return m
    
    @staticmethod
    def to_torch_relu_nn(weights: List[torch.Tensor], biases: List[torch.Tensor]) -> torch.nn.Sequential:
        assert len(weights) == len(biases)
        modules = []
        for w, b in zip(weights, biases):
            modules += [Util.to_torch_linear(w, b), torch.nn.ReLU()]
        modules = modules[:-1]
        return torch.nn.Sequential(*modules)
    
    @staticmethod
    def fill_block(m: torch.Tensor, i: int, j: int, target: torch.Tensor):
        start_row, end_row = target.shape[0] * i, target.shape[1] * (i + 1)
        start_col, end_col = target.shape[0] * j, target.shape[1] * (j + 1)
        m[start_row:end_row, start_col:end_col] = target
        
    @staticmethod
    def make_block_diagonal(block: torch.Tensor, times: int) -> torch.Tensor:
        O = block * 0
        blocks = np.empty((times, times), dtype=object)
        for i in range(times):
            for j in range(times):
                blocks[i, j] = block if i == j else O
        blocks = [torch.cat(tuple(line), dim=1) for line in blocks]
        return torch.cat(tuple(blocks), dim=0)
    
    @staticmethod
    def repeat_tensor(x: torch.Tensor, times: int) -> torch.Tensor:
        return torch.cat((x,) * times, dim=0)
    
    @staticmethod
    def list_round(x, digits: int) -> list:
        if issubclass(type(x), torch.Tensor):
            x = Util.to_numpy(x)
        return [round(y, digits) for y in x]
    
    ### DQNroute-specific:
    
    @staticmethod
    def smooth(p, alpha: float):
        # smoothing to get rid of 0 and 1 probabilities that lead to saturated gradients
        return (1 - alpha) * p + alpha / 2
    
    @staticmethod
    def unsmooth(p, alpha: float):
        return (p - alpha / 2) / (1 - alpha)

    @staticmethod
    def q_values_to_first_probability(qs: torch.Tensor, temperature: float, alpha: float) -> torch.Tensor:
        return Util.smooth((qs / temperature).softmax(dim=0)[0], alpha)
    
    @staticmethod
    def transform_embeddings(sink_embedding    : torch.Tensor,
                             current_embedding : torch.Tensor,
                             neighbor_embedding: torch.Tensor) -> torch.Tensor:
        return torch.cat((sink_embedding     - current_embedding,
                          neighbor_embedding - current_embedding), dim=1)
