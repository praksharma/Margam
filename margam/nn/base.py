from abc import ABC, abstractmethod
from .utils import activation, initialiser
import torch

class BaseNetwork(torch.nn.Module):
    """
    Base class for all neural networks
    """
    def __init__(self) -> None:
        super().__init__() # initialise all methods from nn.Module
        
        self.activation_function = activation
        self.initialiser_function = initialiser        

    @abstractmethod
    def forward(self):
        raise NotImplementedError(f"{self.__class__.__name__} must implement the forward() method.")