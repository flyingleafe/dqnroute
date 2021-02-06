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
    cuda_enabled = False
        
    @staticmethod
    def optimizable_clone(x: torch.Tensor) -> torch.Tensor:
        """
        Clones a PyTorch tensor and makes it suitable for optimization.
        :param x: input tensor.
        :return: x with enabled gradients.
        """
        return Util.conditional_to_cuda(x.clone().detach()).requires_grad_(True)
    
    @staticmethod
    def conditional_to_cuda(x: Union[torch.Tensor, torch.nn.Module]) -> torch.Tensor:
        """
        Returns the tensor/module on GPU if there is at least 1 GPU, otherwise just returns the tensor.
        :param x: a PyTorch tensor or module.
        :return: x on GPU if there is at least 1 GPU, otherwise just x.
        """
        return x.cuda() if (Util.cuda_enabled and torch.cuda.is_available()) else x
    
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
    def to_numpy(x: torch.Tensor) -> np.ndarray:
        """
        Converts a PyTorch tensor to a numpy array.
        :param x: input tensor.
        :return: output numpy array.
        """
        return x.detach().cpu().numpy()

    @staticmethod
    def to_torch_linear(weight: torch.Tensor, bias: torch.Tensor) -> torch.nn.Linear:
        """
        Constructs a torch.nn.Linear layer from the given weight matrix and bias vector.
        :param weight: weight matrix.
        :param bias: bias vector.
        :return: torch.nn.Linear layer.
        """
        m = torch.nn.Linear(*weight.shape)
        m.weight = torch.nn.Parameter(weight)
        m.bias   = torch.nn.Parameter(bias)
        return m
    
    @staticmethod
    def to_torch_relu_nn(weights: List[torch.Tensor], biases: List[torch.Tensor]) -> torch.nn.Sequential:
        """
        Constructs a multiplayer perceptron with ReLUs.
        :param weights: weight matrices.
        :param biases: bias vectors.
        :return multiplayer perceptron as a PyTorch model.
        """
        assert len(weights) == len(biases)
        modules = []
        for w, b in zip(weights, biases):
            modules += [Util.to_torch_linear(w, b), torch.nn.ReLU()]
        modules = modules[:-1]
        return torch.nn.Sequential(*modules)
    
    @staticmethod
    def fill_block(m: torch.Tensor, i: int, j: int, target: torch.Tensor):
        """
        Fills a block of a matrix.
        :param m: matrix to which the block will be written.
        :param i: row index of the block to write.
        :param j: column index of the block to write.
        :param target: the block to write.
        """
        start_row, end_row = target.shape[0] * i, target.shape[1] * (i + 1)
        start_col, end_col = target.shape[0] * j, target.shape[1] * (j + 1)
        m[start_row:end_row, start_col:end_col] = target
        
    @staticmethod
    def make_block_diagonal(block: torch.Tensor, times: int) -> torch.Tensor:
        """
        Constructs a block-diagonal matrix filled with the single given matrix.
        :param block: block to use for filling.
        :param times: number of times to repeat block.
        :return: block-diagonal matrix filled with block.
        """
        O = block * 0
        blocks = np.empty((times, times), dtype=object)
        for i in range(times):
            for j in range(times):
                blocks[i, j] = block if i == j else O
        blocks = [torch.cat(tuple(line), dim=1) for line in blocks]
        return torch.cat(tuple(blocks), dim=0)
    
    @staticmethod
    def repeat_tensor(x: torch.Tensor, times: int) -> torch.Tensor:
        """
        Repeats the given vector several times.
        :param x: vector to repeat.
        :param times: number of times to repeat x.
        :return: repeated x.
        """
        return torch.cat((x,) * times, dim=0)
    
    @staticmethod
    def list_round(x: Iterable, digits: int) -> list:
        """
        Rounds each number in a collection and returns the result.
        :param x: target collection.
        :param digits: number of digits after the period to leave.
        :return: list of rounded elements.
        """
        if issubclass(type(x), torch.Tensor):
            x = Util.to_numpy(x)
        return [round(y, digits) for y in x]
    
    @staticmethod
    def smooth(p, alpha: float):
        """
        Applies probability smoothing, which shifts all probabilities closer to 0.5. This removes
        problems related to dealing with infinite or large values caused by extreme probabilities.
        :param p: number, array or tensor.
        :param alpha: smoothing parameter (between 0 and 1).
        :return: smoothed p.
        """
        # smoothing to get rid of 0 and 1 probabilities that lead to saturated gradients
        return (1 - alpha) * p + alpha / 2
    
    @staticmethod
    def unsmooth(p, alpha: float):
        """
        Reverse of smooth().
        :param p: number, array or tensor.
        :param alpha: smoothing parameter (between 0 and 1).
        :return: x such that smooth(x) = p.
        """
        return (p - alpha / 2) / (1 - alpha)

    @staticmethod
    def q_values_to_first_probability(qs: torch.Tensor, temperature: float, alpha: float) -> torch.Tensor:
        """
        For DQNroute-LE, computes the probability of routing to the first successor of a node based on
        produced Q values.
        :param qs: 1D tensor of Q values for each successor of the current node.
        :param temperature: temperature (T) hyperparameter.
        :param alpha: probability smoothing parameter (between 0 and 1).
        :return: the probability of routing to the first successor of a node.
        """
        return Util.smooth((qs / temperature).softmax(dim=0)[0], alpha)
    
    @staticmethod
    def transform_embeddings(sink_embedding    : torch.Tensor,
                             current_embedding : torch.Tensor,
                             neighbor_embedding: torch.Tensor) -> torch.Tensor:
        """
        For DQNroute-LE, subtracts the embedding of the current node from other embeddings.
        Thus, the embeddings are converted to the inputs of the neural network.
        :param sink_embedding: embedding of the sink.
        :param current_embedding: embedding of the current node.
        :param neighbor_embedding: embedding of the chosen successor of the current node.
        :return input to the neural network of DQNroute-LE.
        """
        return torch.cat((sink_embedding     - current_embedding,
                          neighbor_embedding - current_embedding), dim=1)
