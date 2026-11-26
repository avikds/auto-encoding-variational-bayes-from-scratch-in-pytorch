"""
Auto-Encoding Variational Bayes from Scratch in PyTorch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - GaussianEncoder
import torch
import torch.nn as nn

class GaussianEncoder(nn.Module):
    """q(z | x): tanh hidden layer, followed by linear mean and log variance layers.

    All parameters are initialized from N(0, 0.01^2).
    """

    def __init__(self, x_dim=784, h_dim=500, z_dim=20):
        super().__init__()

        # Define the three fully connected layers.
        self.hidden = nn.Linear(x_dim, h_dim)
        self.mu = nn.Linear(h_dim, z_dim)
        self.logvar = nn.Linear(h_dim, z_dim)

        # Initialize all weights and biases from N(0, 0.01^2).
        for layer in (self.hidden, self.mu, self.logvar):
            nn.init.normal_(layer.weight, mean=0.0, std=0.01)
            nn.init.normal_(layer.bias, mean=0.0, std=0.01)

    def forward(self, x):
        """Return (mu, logvar), each with shape (B, z_dim)."""

        # Transform the input through the hidden layer with tanh activation.
        h = torch.tanh(self.hidden(x))

        # Compute the mean and log variance of the diagonal Gaussian.
        mu = self.mu(h)
        logvar = self.logvar(h)

        return mu, logvar

