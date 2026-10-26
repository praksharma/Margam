import torch
from .base import BaseNetwork

class FullyConnected(BaseNetwork):
    """
    Implementation of Fully Connected neural network
    """
    def __init__(self, 
                layer_size : list,
                activation : str,
                initialiser : str,
                input_bounds = None,
                ):
        super().__init__()

        # Validate the types of the attributes
        if not isinstance(layer_size, list):
            raise TypeError("The 'layer_size' attribute must be an list.")
        if not isinstance(activation, str):
            raise TypeError("The 'activation' attribute must be a string.")
        if not isinstance(initialiser, str):
            raise TypeError("The 'initialiser' attribute must be a string.")
        
        self.activation = self.activation_function(activation)
        self.layer_size = layer_size
        self.initialiser = self.initialiser_function(initialiser)

        # Fixed bounds are independent of the evaluated batch, preserving PDE
        # derivatives and making single-point prediction well-defined.
        lower = upper = None
        if input_bounds is not None:
            lower = torch.as_tensor(input_bounds[0], dtype=torch.get_default_dtype())
            upper = torch.as_tensor(input_bounds[1], dtype=torch.get_default_dtype())
            if lower.shape != upper.shape or lower.ndim > 1:
                raise ValueError("Bounds must have matching scalar or vector shapes.")
            if lower.ndim == 1 and len(lower) != layer_size[0]:
                raise ValueError("Bounds must match the network input dimension.")
            if not torch.isfinite(lower).all() or not torch.isfinite(upper).all():
                raise ValueError("Bounds must be finite.")
            if not torch.all(upper > lower):
                raise ValueError("Upper bounds must exceed lower bounds.")
        self.register_buffer("input_lower", lower)
        self.register_buffer("input_upper", upper)

        # Initialise neural network as a list using nn.Modulelist 
        self.linears = torch.nn.ModuleList([torch.nn.Linear(self.layer_size[i], self.layer_size[i+1]) for i in range(0,len(self.layer_size)-1)])

        # initialise the weights
        self.weight_init()


    def weight_init(self):
        # weight initialisation
        for i in range(len(self.layer_size)-1):
            
            # weights from a normal distribution with 
            # Recommended gain value for tanh = 5/3?
            self.initialiser(self.linears[i].weight.data, gain=1.0)
            
            # set biases to zero
            # torch.nn.init.zeros_(self.linears[i].bias.data)
            torch.nn.init.constant_(self.linears[i].bias.data, 0.01)

    def forward(self, input):
        """
        Implementation of forward for the Fully Connected network.
        All we need is the inputs tensor and rest of the things are already in the class.
        """

        if self.input_lower is not None:
            input = (input - self.input_lower) / (self.input_upper - self.input_lower)
        value = input
        for layer in self.linears[:-1]:
            value = self.activation(layer(value))
        return self.linears[-1](value)



class TestNetwork(BaseNetwork, torch.nn.Module):
    """
    Test network for debugging.
    """
    def __init__(self):
        super().__init__()

    def forward(self):
        print(f"Forward pass of {self.__class__.__name__} module.")
