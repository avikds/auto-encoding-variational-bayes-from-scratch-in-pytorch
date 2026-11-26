# Auto-Encoding Variational Bayes from Scratch in PyTorch

Auto-Encoding Variational Bayes (Kingma and Welling, 2013), rebuilt in PyTorch on MNIST. Write the paper's variational auto-encoder: a Gaussian MLP encoder that outputs a mean and a log variance, a Bernoulli MLP decoder, both with one tanh hidden layer and weights drawn with standard deviation 0.01, and the reparameterization z = mu + sigma * eps. Build both SGVB estimators of the variational lower bound, the generic estimator A and estimator B with the KL term in closed form, and measure their gradient variance against the naive Monte Carlo estimator the paper calls impractical. Train with the AEVB algorithm (minibatches of 100, one sample per datapoint, Adagrad, a Gaussian prior on the weights) and with the wake-sleep baseline on the same encoder and decoder, and compare their lower bounds for 3, 20 and 200 latent variables. Finish with the paper's other experiments at CPU size: the learned 2-D latent manifold, and the marginal likelihood estimated with Hybrid Monte Carlo for small and large training sets.

## How to run

```bash
python scaffold.py
```

## Steps

- [x] **1.** GaussianEncoder
- [x] **2.** BernoulliDecoder
- [x] **3.** reparameterize
- [x] **4.** log_normal_diag
- [x] **5.** sgvb_a
- [x] **6.** neg_kl_gaussian
- [x] **7.** sgvb_b
- [x] **8.** gradient_variance
- [x] **9.** aevb_step
- [x] **10.** wake_sleep_step
- [x] **11.** train_model
- [x] **12.** latent_manifold
- [x] **13.** hmc_posterior
- [x] **14.** marginal_likelihood

---

Built on Deep-ML.
