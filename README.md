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

## Results

```
MNIST: 50000 training images, 10000 test images, intensities used as Bernoulli targets
Settings as in the paper: tanh MLPs, weights ~ N(0, 0.01^2), M = 100, L = 1, Adagrad, N(0, I) prior on all weights.
Changed for CPU time: 200 hidden units (paper 500), 2 x 10^5 training points per run (paper up to about 10^8),
latent sizes 3, 20 and 200 only (the paper also runs 5 and 10), no Frey Face, no Monte Carlo EM baseline.

[1] training-set bound after 20,000 points (200 updates), N_z = 20, for each step size in {0.01, 0.02, 0.1}
    aevb       0.01:    -198.8  0.02:    -196.2  0.1:    -225.0   -> step size 0.02
    wake-sleep 0.01:    -200.6  0.02:    -199.6  0.1:  -38299.5   -> step size 0.02

[2] average lower bound per image (nats) after 200,000 training points; paper after about 10^8 in brackets
    N_z = 3   aevb       train  -169.5 [-128]  test  -169.8 [-133]  train minus test   0.3   train bound every 40k points: -191 -177 -169 -167 -168
    N_z = 3   wake-sleep train  -169.0 [-139]  test  -169.2 [-138]  train minus test   0.2   train bound every 40k points: -189 -183 -171 -168 -167
    N_z = 20  aevb       train  -144.7 [-97]  test  -144.4 [-100]  train minus test  -0.3   train bound every 40k points: -179 -163 -154 -149 -144
    N_z = 20  wake-sleep train  -164.4 [-104]  test  -164.5 [-104]  train minus test   0.1   train bound every 40k points: -190 -180 -172 -169 -163
    N_z = 200 aevb       train  -152.3 [-97]  test  -152.1 [-100]  train minus test  -0.2   train bound every 40k points: -191 -174 -165 -159 -151
    N_z = 200 wake-sleep train  -171.0 [-103]  test  -171.0 [-103]  train minus test   0.0   train bound every 40k points: -198 -187 -177 -171 -169
    N_z = 3   AEVB minus wake-sleep on test images: -0.7 nats
    N_z = 20  AEVB minus wake-sleep on test images: +20.1 nats
    N_z = 200 AEVB minus wake-sleep on test images: +18.9 nats

[3] total variance of the encoder gradient over 30 noise draws, 100 training images
    untrained N_z = 20      naive  3.74e+09   A  1.27e+04   B     0.942   naive / B    4e+09   A / B 13509.13
    AEVB-trained N_z = 20   naive     6e+10   A  3.74e+06   B  4.15e+06   naive / B  1.4e+04   A / B   0.90
    AEVB-trained N_z = 200  naive  1.25e+11   A   7.8e+06   B  1.04e+07   naive / B  1.2e+04   A / B   0.75

[4] estimated log p(x) per image (HMC estimator, first 100 training and first 100 test images), N_z = 3, 100 hidden units, 100,000 training points; paper's values at the end of its runs in brackets
    N_train = 1000   aevb       log p(x): train  -168.5 [-109]  test  -177.0 [-143]   lower bound: train  -169.7  test  -178.4
    N_train = 1000   wake-sleep log p(x): train  -172.8 [-125.5]  test  -184.1 [-145]   lower bound: train  -174.3  test  -185.4
    N_train = 50000  aevb       log p(x): train  -179.7 [-131]  test  -180.0 [-138]   lower bound: train  -181.7  test  -182.6
    N_train = 50000  wake-sleep log p(x): train  -176.5 [-137]  test  -176.7 [-141]   lower bound: train  -179.0  test  -179.5

[5] AEVB with a 2-D latent space, 200,000 training points
    test bound -177.4; 20 x 20 decoded grid, digit class of each image's nearest training image:
    0: 134  1: 28  2: 14  3: 4  4: 72  5: 1  6: 0  7: 36  8: 45  9: 66   (9 of 10 classes appear)
    the encoder's 2-D codes keep the classes apart: nearest-neighbour accuracy 35.6% on 2,000 test codes (chance 10%)
```
