"""Neural Networks base class."""

import torch


class NeuralNetwork(torch.nn.Module):
    """Base class for neural networks."""

    def __init__(self, num_inputs: int, num_outputs: int):
        """Initialize the neural network.

        Args:
            num_inputs (int): Number of input features.
            num_outputs (int): Number of output features.
        """
        super().__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs

        self.layers = torch.nn.Sequential(
            # 1st hidden layer
            torch.nn.Linear(num_inputs, 30),
            torch.nn.ReLU(),
            # 2nd hidden layer
            torch.nn.Linear(30, 20),
            torch.nn.ReLU(),
            # Output layer
            torch.nn.Linear(20, num_outputs),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the neural network.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        logits = self.layers(x)
        return logits
