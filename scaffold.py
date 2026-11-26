"""
Auto-Encoding Variational Bayes from Scratch in PyTorch scaffold.

Run this with: python scaffold.py
Uses functions defined in model.py.
"""

from model import *  # noqa: F401, F403 (pulls in your solution functions)

"""Auto-Encoding Variational Bayes from Scratch in PyTorch (Kingma and Welling, 2013).

Story: train the paper's variational auto-encoder on MNIST with AEVB and the same networks
with wake-sleep; pick each method's Adagrad step size from the paper's set; compare their
lower bounds for 3, 20 and 200 latent variables; measure the gradient variance of the naive
estimator and of SGVB estimators A and B; compare the methods on the marginal likelihood
estimated with Hybrid Monte Carlo, for 1,000 and 50,000 training images; and decode the
learned 2-D latent manifold. Everything runs on a CPU in a few minutes, so every run is
1,000 to 2,000 updates long where the paper's runs are hundreds of times longer: the paper's
numbers are printed next to ours as a reference for direction, not for size.
"""
import os
import math
import tempfile
import urllib.request
import numpy as np
import torch


def load_mnist(n_train=50000, n_test=10000):
    """The first n_train MNIST training images and the first n_test test images, flattened to [0, 1]."""
    path = os.path.join(tempfile.gettempdir(), "dml_mnist.npz")
    if not os.path.exists(path):
        part = path + f".{os.getpid()}.part"
        urllib.request.urlretrieve("https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz", part)
        os.replace(part, path)
    with np.load(path) as d:
        x_tr, y_tr, x_te, y_te = d["x_train"][:n_train], d["y_train"][:n_train], d["x_test"][:n_test], d["y_test"][:n_test]
    images = lambda a: torch.tensor(a.reshape(len(a), -1), dtype=torch.float32) / 255.0
    labels = lambda a: torch.tensor(a.astype(np.int64))
    return images(x_tr), labels(y_tr), images(x_te), labels(y_te)


# Approximate values read off the paper's plots (lower bound after about 10^8 training points;
# marginal likelihood at the end of its runs). Only good to about +/- 2 nats.
PAPER_BOUND = {3: ((-128, -133), (-139, -138)), 20: ((-97, -100), (-104, -104)), 200: ((-97, -100), (-103, -103))}
PAPER_LOGP = {1000: ((-109, -143), (-125.5, -145)), 50000: ((-131, -138), (-137, -141))}


def main():
    X, y, X_test, y_test = load_mnist(50000, 10000)
    print(f"MNIST: {len(X)} training images, {len(X_test)} test images, intensities used as Bernoulli targets")
    print("Settings as in the paper: tanh MLPs, weights ~ N(0, 0.01^2), M = 100, L = 1, Adagrad, N(0, I) prior on all weights.")
    print("Changed for CPU time: 200 hidden units (paper 500), 2 x 10^5 training points per run (paper up to about 10^8),")
    print("latent sizes 3, 20 and 200 only (the paper also runs 5 and 10), no Frey Face, no Monte Carlo EM baseline.")
    n_points = 200000

    # ---- 1. The Adagrad step size, chosen from the paper's set on the first 200 updates ----
    print("\n[1] training-set bound after 20,000 points (200 updates), N_z = 20, for each step size in {0.01, 0.02, 0.1}")
    lr = {}
    for method in ("aevb", "wake-sleep"):
        scores = {}
        for eta in (0.01, 0.02, 0.1):
            scores[eta] = train_model(method, X, X_test, z_dim=20, n_points=20000, lr=eta, eval_every=20000)[2][-1][1]
        lr[method] = max(scores, key=scores.get)
        print(f"    {method:<10} " + "  ".join(f"{eta}: {v:9.1f}" for eta, v in scores.items()) + f"   -> step size {lr[method]}")

    # ---- 2. Lower bound: AEVB against wake-sleep for 3, 20 and 200 latent variables ----
    print(f"\n[2] average lower bound per image (nats) after {n_points:,} training points; paper after about 10^8 in brackets")
    models = {}
    for z_dim in (3, 20, 200):
        for method in ("aevb", "wake-sleep"):
            enc, dec, hist = train_model(method, X, X_test, z_dim=z_dim, n_points=n_points, lr=lr[method], eval_every=n_points // 5)
            models[(method, z_dim)] = (enc, dec)
            train, test = average_bound(enc, dec, X[:10000]), average_bound(enc, dec, X_test)
            paper = PAPER_BOUND[z_dim][0 if method == "aevb" else 1]
            curve = " ".join(f"{h[1]:.0f}" for h in hist)
            print(f"    N_z = {z_dim:<3} {method:<10} train {train:7.1f} [{paper[0]}]  test {test:7.1f} [{paper[1]}]"
                  f"  train minus test {train - test:5.1f}   train bound every 40k points: {curve}")
    for z_dim in (3, 20, 200):
        a = average_bound(*models[("aevb", z_dim)], X_test)
        w = average_bound(*models[("wake-sleep", z_dim)], X_test)
        print(f"    N_z = {z_dim:<3} AEVB minus wake-sleep on test images: {a - w:+.1f} nats")

    # ---- 3. Gradient variance of the three estimators ----
    print("\n[3] total variance of the encoder gradient over 30 noise draws, 100 training images")
    torch.manual_seed(0)
    untrained = (GaussianEncoder(784, 200, 20), BernoulliDecoder(20, 200, 784))
    for name, (enc, dec) in (("untrained N_z = 20", untrained), ("AEVB-trained N_z = 20", models[("aevb", 20)]),
                             ("AEVB-trained N_z = 200", models[("aevb", 200)])):
        torch.manual_seed(0)
        var = {e: gradient_variance(enc, dec, X[:100], e, n_draws=30) for e in ("naive", "A", "B")}
        print(f"    {name:<23} naive {var['naive']:9.3g}   A {var['A']:9.3g}   B {var['B']:9.3g}"
              f"   naive / B {var['naive'] / var['B']:8.2g}   A / B {var['A'] / var['B']:6.2f}")

    # ---- 4. Marginal likelihood with Hybrid Monte Carlo: 100 hidden units, 3 latent variables ----
    print("\n[4] estimated log p(x) per image (HMC estimator, first 100 training and first 100 test images), N_z = 3,"
          " 100 hidden units, 100,000 training points; paper's values at the end of its runs in brackets")
    for n_train in (1000, 50000):
        for method in ("aevb", "wake-sleep"):
            enc, dec, _ = train_model(method, X[:n_train], X_test, z_dim=3, h_dim=100, n_points=100000, lr=lr[method], eval_every=100000)
            torch.manual_seed(0)
            logp = marginal_likelihood(dec, torch.cat([X[:100], X_test[:100]]))
            bound_tr, bound_te = average_bound(enc, dec, X[:100]), average_bound(enc, dec, X_test[:100])
            paper = PAPER_LOGP[n_train][0 if method == "aevb" else 1]
            print(f"    N_train = {n_train:<6} {method:<10} log p(x): train {float(logp[:100].mean()):7.1f} [{paper[0]}]"
                  f"  test {float(logp[100:].mean()):7.1f} [{paper[1]}]   lower bound: train {bound_tr:7.1f}  test {bound_te:7.1f}")

    # ---- 5. The learned 2-D latent manifold ----
    print(f"\n[5] AEVB with a 2-D latent space, {n_points:,} training points")
    enc, dec, hist = train_model("aevb", X, X_test, z_dim=2, n_points=n_points, lr=lr["aevb"], eval_every=n_points)
    imgs = latent_manifold(dec, n=20)
    nearest = y[:10000][torch.cdist(imgs, X[:10000]).argmin(dim=1)]
    counts = torch.bincount(nearest, minlength=10).tolist()
    print(f"    test bound {average_bound(enc, dec, X_test):.1f}; 20 x 20 decoded grid, digit class of each image's nearest training image:")
    print("    " + "  ".join(f"{d}: {c}" for d, c in enumerate(counts)) + f"   ({sum(c > 0 for c in counts)} of 10 classes appear)")
    with torch.no_grad():
        codes_tr, codes_te = enc(X[:10000])[0], enc(X_test[:2000])[0]
    knn = y[:10000][torch.cdist(codes_te, codes_tr).argmin(dim=1)]
    print(f"    the encoder's 2-D codes keep the classes apart: nearest-neighbour accuracy {float((knn == y_test[:2000]).float().mean()):.1%}"
          " on 2,000 test codes (chance 10%)")


if __name__ == "__main__":
    main()

