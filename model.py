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

# Step 2 - BernoulliDecoder
import torch.nn.functional as F


class BernoulliDecoder(nn.Module):
    """p(x | z): y = sigmoid(W2 tanh(W1 z + b1) + b2).

    All parameters are initialized from N(0, 0.01^2).
    """

    def __init__(self, z_dim=20, h_dim=500, x_dim=784):
        super().__init__()

        # Define the hidden and output fully connected layers.
        self.hidden = nn.Linear(z_dim, h_dim)
        self.out = nn.Linear(h_dim, x_dim)

        # Initialize all weights and biases from N(0, 0.01^2).
        for layer in (self.hidden, self.out):
            nn.init.normal_(layer.weight, mean=0.0, std=0.01)
            nn.init.normal_(layer.bias, mean=0.0, std=0.01)

    def logits(self, z):
        # Apply the hidden linear transformation and tanh activation.
        h = torch.tanh(self.hidden(z))

        # Return the raw pixel logits without applying sigmoid.
        return self.out(h)

    def forward(self, z):
        """Return pixel probabilities y."""

        # Convert logits into Bernoulli probabilities.
        return torch.sigmoid(self.logits(z))

    def log_prob(self, x, z):
        """Return the Bernoulli log probability, summed over pixels."""

        # Compute raw logits for numerically stable likelihood evaluation.
        logits = self.logits(z)

        # Expand x to support both (B, z_dim) and (L, B, z_dim) inputs.
        x = x.expand_as(logits)

        # Compute the negative binary cross-entropy and sum over pixels.
        log_likelihood = -F.binary_cross_entropy_with_logits(
            logits,
            x,
            reduction="none"
        )

        return log_likelihood.sum(dim=-1)

