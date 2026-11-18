import torch

"""
This file contains helper functions common to all neural networks
"""

_ACTIVATION_FACTORIES = {
    "linear": torch.nn.Identity,
    "tanh": torch.nn.Tanh,
    "relu": torch.nn.ReLU,
}

activation_dict = {name: factory() for name, factory in _ACTIVATION_FACTORIES.items()}

initialiser_dict = {
    "Glorot uniform": torch.nn.init.xavier_uniform_,
    "Xavier normal": torch.nn.init.xavier_normal_,
}


def activation(activation_string):
    """
    Returns a fresh activation function module.
    """
    if activation_string in _ACTIVATION_FACTORIES:
        return _ACTIVATION_FACTORIES[activation_string]()
    if activation_string in activation_dict:
        return activation_dict[activation_string]
    raise ValueError("Invalid activation function.")


def initialiser(intialiser_string):
    """
    Returns the initialiser callable.
    """
    if intialiser_string in initialiser_dict:
        return initialiser_dict[intialiser_string]
    raise ValueError("Invalid initialiser.")