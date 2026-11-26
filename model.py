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

# Step 8 - gradient_variance
def gradient_variance(encoder, decoder, x, estimator, n_draws=50):
    """Measure the total variance of the encoder gradients across
    independent Monte Carlo draws.

    Supported estimators:
        "A": Generic SGVB estimator.
        "B": SGVB estimator with analytical KL divergence.
        "naive": Naive Monte Carlo score-function estimator.

    Args:
        encoder: GaussianEncoder instance.
        decoder: BernoulliDecoder instance.
        x: Input batch of shape (B, D).
        estimator: One of "A", "B", or "naive".
        n_draws: Number of independent gradient estimates.

    Returns:
        Total gradient variance as a Python float.
    """

    # Validate the estimator selection.
    if estimator not in ("A", "B", "naive"):
        raise ValueError("estimator must be 'A', 'B', or 'naive'")

    if n_draws < 2:
        raise ValueError("n_draws must be at least 2")

    # Collect the encoder parameters whose gradient variance is measured.
    params = list(encoder.parameters())

    # Store one flattened gradient vector for each independent draw.
    gradients = []

    for _ in range(n_draws):

        if estimator == "A":
            # Generic SGVB estimator: sample all terms.
            obj = sgvb_a(encoder, decoder, x).sum()

        elif estimator == "B":
            # SGVB estimator with the analytical KL divergence.
            obj = sgvb_b(encoder, decoder, x).sum()

        else:
            # Naive Monte Carlo score-function estimator.

            # Compute the approximate posterior parameters.
            mu, logvar = encoder(x)

            # Sample a latent vector and detach it to remove
            # the gradient path through the reparameterized sample.
            eps = torch.randn_like(mu)
            z = reparameterize(mu, logvar, eps).detach()

            # Compute the log probability under q(z | x).
            log_q = log_normal_diag(z, mu, logvar)

            # Compute the prior log probability under N(0, I).
            zero = torch.zeros_like(z)
            log_pz = log_normal_diag(z, zero, zero)

            # Compute the learning signal and detach it.
            f = (
                decoder.log_prob(x, z)
                + log_pz
                - log_q
            ).detach()

            # Form the scalar score-function objective.
            obj = (f * log_q).sum()

        # Compute gradients with respect to encoder parameters.
        grads = torch.autograd.grad(obj, params)

        # Flatten and concatenate all parameter gradients into one vector.
        grad_vector = torch.cat([
            grad.reshape(-1) for grad in grads
        ])

        gradients.append(grad_vector)

    # Stack gradient vectors into a tensor of shape (n_draws, P).
    gradients = torch.stack(gradients, dim=0)

    # Calculate the unbiased variance of each gradient coordinate.
    coordinate_variance = gradients.var(dim=0)

    # Return the sum of coordinate-wise variances as a Python float.
    return coordinate_variance.sum().item()

# Step 9 - aevb_step
def aevb_step(encoder, decoder, optimizer, xb, N, estimator="B"):
    """Perform one minibatch AEVB update using gradient ascent.

    The objective consists of:
        1. The minibatch estimate of the full training-set lower bound.
        2. A standard Gaussian prior on all encoder and decoder parameters.

    Args:
        encoder: GaussianEncoder instance.
        decoder: BernoulliDecoder instance.
        optimizer: Optimizer containing both networks' parameters.
        xb: Minibatch of training data, shape (M, D).
        N: Total number of training datapoints.
        estimator: "A" or "B", default is "B".

    Returns:
        Mean lower-bound estimate per datapoint before the update.
    """

    # Determine the minibatch size.
    M = len(xb)

    if M == 0:
        raise ValueError("The minibatch must contain at least one datapoint.")

    if N <= 0 or N < M:
        raise ValueError("N must be positive and at least as large as M.")

    if estimator not in ("A", "B"):
        raise ValueError("estimator must be 'A' or 'B'.")

    # Compute the per-datapoint variational lower-bound estimates.
    if estimator == "A":
        bound = sgvb_a(encoder, decoder, xb)
    else:
        bound = sgvb_b(encoder, decoder, xb)

    # Scale the minibatch sum to estimate the full training-set bound.
    scaled_bound = (N / M) * bound.sum()

    # Apply a standard Gaussian prior to every parameter
    # of both the encoder and decoder.
    weight_prior = sum(
        param.pow(2).sum()
        for module in (encoder, decoder)
        for param in module.parameters()
    )

    # Construct the MAP objective, omitting the constant term.
    objective = scaled_bound - 0.5 * weight_prior

    # PyTorch optimizers perform gradient descent.
    # Negating the objective therefore implements gradient ascent.
    optimizer.zero_grad()
    (-objective).backward()
    optimizer.step()

    # Return the average lower-bound estimate before the update.
    return float(bound.mean())

# Step 10 - wake_sleep_step
def wake_sleep_step(encoder, decoder, optimizer, xb, N):
    """Perform one wake-sleep update using the same encoder and decoder.

    The wake phase trains the decoder using encoder-generated latent codes.
    The sleep phase trains the encoder using samples generated by the decoder.

    Args:
        encoder: GaussianEncoder instance.
        decoder: BernoulliDecoder instance.
        optimizer: Optimizer containing both networks' parameters.
        xb: Minibatch of training data, shape (M, D).
        N: Total number of training datapoints.

    Returns:
        None.
    """

    # Determine the minibatch size.
    M = len(xb)

    # Draw all random quantities without tracking gradients.
    # Follow the exact sampling order specified in the project.
    with torch.no_grad():

        # 1. Encode the real minibatch and sample wake latent codes.
        mu, logvar = encoder(xb)
        z = reparameterize(mu, logvar, torch.randn_like(mu))

        # 2. Sample dream latent codes from the standard normal prior.
        z_dream = torch.randn_like(mu)

        # 3. Generate binary dream images using the decoder.
        x_dream = torch.bernoulli(decoder(z_dream))

    # Wake phase: train the decoder to reconstruct real images
    # from latent codes sampled by the encoder.
    wake = decoder.log_prob(xb, z)

    # Sleep phase: train the encoder to recover the latent codes
    # that generated the dream images.
    mu_d, logvar_d = encoder(x_dream)
    sleep = log_normal_diag(z_dream, mu_d, logvar_d)

    # Estimate the full training-set objective using the minibatch.
    wake_sleep_bound = (N / M) * (wake.sum() + sleep.sum())

    # Apply a standard Gaussian prior to every parameter
    # of both the encoder and decoder.
    weight_prior = sum(
        param.pow(2).sum()
        for module in (encoder, decoder)
        for param in module.parameters()
    )

    # Construct the MAP objective, omitting the constant term.
    objective = wake_sleep_bound - 0.5 * weight_prior

    # Perform gradient ascent using the optimizer.
    optimizer.zero_grad()
    (-objective).backward()
    optimizer.step()

    # This function intentionally returns None.

# Step 11 - train_model
def average_bound(encoder, decoder, X, seed=0):
    """Compute the average variational lower bound over X.

    Evaluation is performed without gradients and without changing
    the global PyTorch random number generator state.

    Args:
        encoder: GaussianEncoder instance.
        decoder: BernoulliDecoder instance.
        X: Input dataset.
        seed: Random seed used for reproducible evaluation.

    Returns:
        Average lower-bound estimate as a Python float.
    """

    # Save the current global random number generator state.
    rng_state = torch.get_rng_state()

    try:
        # Set the seed for deterministic evaluation.
        torch.manual_seed(seed)

        # Evaluate the lower bound without computing gradients.
        with torch.no_grad():
            bound = sgvb_b(encoder, decoder, X).mean()

        return float(bound.item())

    finally:
        # Restore the original random number generator state.
        # Evaluation therefore does not affect subsequent training.
        torch.set_rng_state(rng_state)


def train_model(
    method,
    X,
    X_test,
    z_dim=20,
    h_dim=200,
    n_points=100000,
    lr=0.02,
    eval_every=20000,
    seed=0
):
    """Train a VAE using AEVB or the wake-sleep algorithm.

    Training uses minibatches of 100 datapoints and Adagrad optimization.

    Args:
        method: Training algorithm, either "aevb" or "wake-sleep".
        X: Training dataset.
        X_test: Test dataset.
        z_dim: Number of latent dimensions.
        h_dim: Number of hidden units.
        n_points: Number of training points to evaluate.
        lr: Adagrad learning rate.
        eval_every: Evaluation interval.
        seed: Random seed.

    Returns:
        encoder: Trained GaussianEncoder.
        decoder: Trained BernoulliDecoder.
        history: List of (count, train_bound, test_bound) tuples.
    """

    # Validate the training method and evaluation interval.
    if method not in ("aevb", "wake-sleep"):
        raise ValueError("method must be 'aevb' or 'wake-sleep'")

    if eval_every <= 0:
        raise ValueError("eval_every must be positive")

    # Set the random seed for reproducible initialization and training.
    torch.manual_seed(seed)

    # Determine the input dimensionality.
    D = X.shape[1]

    # Initialize a fresh Gaussian encoder.
    encoder = GaussianEncoder(
        x_dim=D,
        h_dim=h_dim,
        z_dim=z_dim
    )

    # Initialize a fresh Bernoulli decoder.
    decoder = BernoulliDecoder(
        z_dim=z_dim,
        h_dim=h_dim,
        x_dim=D
    )

    # Create one Adagrad optimizer for both networks.
    optimizer = torch.optim.Adagrad(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=lr,
        eps=1e-8
    )

    # Initialize the training history and point counter.
    history = []
    count = 0

    # Continue training until the requested number of points is reached.
    while count < n_points:

        # Randomly sample a minibatch of 100 training datapoints.
        indices = torch.randint(0, len(X), (100,))
        xb = X[indices]

        # Perform one update using the selected training algorithm.
        if method == "aevb":

            aevb_step(
                encoder,
                decoder,
                optimizer,
                xb,
                N=len(X)
            )

        else:

            wake_sleep_step(
                encoder,
                decoder,
                optimizer,
                xb,
                N=len(X)
            )

        # Update the number of training points evaluated.
        count += 100

        # Evaluate and record the bounds at the specified intervals.
        if count % eval_every == 0:

            # Evaluate the first 1000 training datapoints.
            train_bound = average_bound(
                encoder,
                decoder,
                X[:1000]
            )

            # Evaluate the first 1000 test datapoints.
            test_bound = average_bound(
                encoder,
                decoder,
                X_test[:1000]
            )

            # Store the evaluation results.
            history.append((
                count,
                train_bound,
                test_bound
            ))

    # Return both trained networks and the evaluation history.
    return encoder, decoder, history

# Step 12 - latent_manifold
def latent_manifold(decoder, n=20):
    """Generate the decoded n x n latent manifold.

    Latent coordinates are evenly spaced in probability under
    the standard normal prior using the inverse Gaussian CDF.

    Args:
        decoder: Trained BernoulliDecoder instance.
        n: Number of grid points along each latent dimension.

    Returns:
        Decoded Bernoulli probabilities of shape (n*n, D).
    """

    # Generate probability coordinates strictly inside (0, 1).
    u = (torch.arange(n, dtype=torch.get_default_dtype()) + 0.5) / n

    # Map probability coordinates through the inverse CDF
    # of the standard normal distribution.
    normal = torch.distributions.Normal(0.0, 1.0)
    z_values = normal.icdf(u)

    # Construct the latent grid in row-major order.
    # The first latent coordinate varies across rows.
    z1 = z_values.repeat_interleave(n)

    # The second latent coordinate varies across columns.
    z2 = z_values.repeat(n)

    # Combine both coordinates into an (n*n, 2) latent grid.
    grid = torch.stack((z1, z2), dim=-1)

    # Decode all latent points without computing gradients.
    with torch.no_grad():
        return decoder(grid)

