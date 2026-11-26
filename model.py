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

# Step 3 - reparameterize
def reparameterize(mu, logvar, eps):
    """Generate a latent sample using the reparameterization trick.

    Args:
        mu: Mean of the Gaussian distribution, shape (B, J).
        logvar: Log variance of the Gaussian distribution, shape (B, J).
        eps: Standard normal noise, shape (B, J) or (L, B, J).

    Returns:
        z: Reparameterized latent sample, broadcast to eps's shape.
    """

    # Convert log variance into standard deviation.
    sigma = torch.exp(0.5 * logvar)

    # Generate the latent sample using the reparameterization trick.
    # PyTorch broadcasting supports both single and multiple samples.
    z = mu + sigma * eps

    return z

# Step 4 - log_normal_diag
import math


def log_normal_diag(z, mu, logvar):
    """Compute the log density of a diagonal Gaussian distribution.

    The result is summed over the last dimension and supports broadcasting.

    Args:
        z: Sampled latent variables.
        mu: Mean of the Gaussian distribution.
        logvar: Log variance of the Gaussian distribution.

    Returns:
        Log probability summed over the last dimension.
    """

    # Compute the squared difference between the sample and the mean.
    squared_diff = (z - mu) ** 2

    # Compute the log probability for each latent dimension.
    log_prob = (
        math.log(2.0 * math.pi)
        + logvar
        + squared_diff * torch.exp(-logvar)
    )

    # Sum over the last dimension and apply the Gaussian normalization factor.
    return -0.5 * log_prob.sum(dim=-1)

# Step 5 - sgvb_a
def sgvb_a(encoder, decoder, x, L=1):
    """Generic SGVB estimator.

    Computes the average variational lower bound over L
    reparameterized latent samples for each datapoint.

    Args:
        encoder: GaussianEncoder instance.
        decoder: BernoulliDecoder instance.
        x: Input batch of shape (B, D).
        L: Number of latent samples per datapoint.

    Returns:
        Lower-bound estimates of shape (B,).
    """

    # Run the encoder once to obtain the Gaussian parameters.
    mu, logvar = encoder(x)

    # Draw L independent standard normal noise samples.
    eps = torch.randn((L,) + tuple(mu.shape))

    # Generate latent samples using the reparameterization trick.
    # Resulting shape: (L, B, z_dim).
    z = reparameterize(mu, logvar, eps)

    # Compute the reconstruction log probability.
    # Resulting shape: (L, B).
    log_px_z = decoder.log_prob(x, z)

    # Compute the standard normal prior log probability.
    zero = torch.zeros_like(z)
    log_pz = log_normal_diag(z, zero, zero)

    # Compute the encoder's approximate posterior log probability.
    log_qz_x = log_normal_diag(z, mu, logvar)

    # Evaluate the generic SGVB estimator for every sample.
    lower_bound = log_px_z + log_pz - log_qz_x

    # Average over the L samples, retaining one value per datapoint.
    return lower_bound.mean(dim=0)

# Step 6 - neg_kl_gaussian
def neg_kl_gaussian(mu, logvar):
    """Compute the negative KL divergence between a diagonal Gaussian
    and the standard normal prior N(0, I).

    Supports arbitrary leading dimensions and sums over the last dimension.

    Args:
        mu: Mean of the Gaussian distribution, shape (..., J).
        logvar: Log variance of the Gaussian distribution, shape (..., J).

    Returns:
        Negative KL divergence, shape (...,).
    """

    # Compute the negative KL divergence for each latent dimension.
    neg_kl = 0.5 * (
        1.0
        + logvar
        - mu.pow(2)
        - torch.exp(logvar)
    )

    # Sum across the latent dimensions.
    return neg_kl.sum(dim=-1)

# Step 7 - sgvb_b
def sgvb_b(encoder, decoder, x, L=1):
    """Estimator B: analytical negative KL divergence plus
    the average reconstruction log probability over L samples.

    Args:
        encoder: GaussianEncoder instance.
        decoder: BernoulliDecoder instance.
        x: Input batch of shape (B, D).
        L: Number of latent samples per datapoint.

    Returns:
        Variational lower-bound estimates of shape (B,).
    """

    # Run the encoder once to obtain the mean and log variance.
    mu, logvar = encoder(x)

    # Draw L independent standard normal noise samples.
    eps = torch.randn((L,) + tuple(mu.shape))

    # Generate latent samples using the reparameterization trick.
    z = reparameterize(mu, logvar, eps)

    # Compute the reconstruction log probability for each sample.
    # Shape: (L, B).
    log_px_z = decoder.log_prob(x, z)

    # Average reconstruction log probabilities over L samples.
    # Shape: (B,).
    reconstruction = log_px_z.mean(dim=0)

    # Compute the analytical negative KL divergence.
    # Shape: (B,).
    kl_term = neg_kl_gaussian(mu, logvar)

    # Combine both terms to obtain the variational lower bound.
    return kl_term + reconstruction

