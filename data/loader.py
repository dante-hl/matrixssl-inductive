# %%
import torch
import numpy as np

# %%
# ignoring labelling: str = 'single', for now, add more if you need
def generate_cube_data(n: int = 50000, v: int = 12500, d: int = 50, k: int = 10, weights=None):
    """
    Generates `n` points in the `d`-dimensional hypercube {-1, 1}^d, with a train-val split of n-v:v.

    Training points are sampled uniformly in the d-dimensional hypercube {-1, 1}^d. Augmentations are produced from these sampled points by independently scaling each of the last d-k dimensions by a uniformly sampled constant within (0, 1]. The first `k` and remaining d-k dimensions correspond to invariant and spurious features respectively.Validation points are sampled uniformly in {-1, 1}^d, with no augmentations.

    n: total number of data (train + val)
    v: number of validation points
    d: total dimension of each datum
    k: number of invariant features
    weights: (k,) array of weights for generating data, sampled from N(0, 1) if None

    Returns train-val data tuple (x1, x2, y), (val_x, val_y)
    Training set: x1, x2, (n-v, d) arrays of augmented data, and labels y (n-v,).
    Validation set: val_x (v, d), val_y (v,)
    """
    assert k < d  # can it be equal..?
    natural = 2 * torch.rand((n, d)) - 1  # uniform in (-1, 1)
    val_x = natural[-v:, :]  # (v, d)
    train_x = natural[:n - v, :]  # (n-v, d)
    print(train_x.shape)
    tau1, tau2 = torch.ones_like(train_x), torch.ones_like(train_x)
    tau1[:, k:] *= torch.rand((n - v, d - k))
    tau2[:, k:] *= torch.rand((n - v, d - k))
    x1, x2 = train_x * tau1, train_x * tau2

    true_w = torch.zeros(d)
    if weights:
        assert weights.shape[0] == k
        true_w[:k] = weights
    else: 
        true_w[:k] = torch.randn(k)
    # print(f"weights: {true_w}")

    # change from {-1, 1} to {0, 1} labels - for linear classification loss
    y = (torch.sign(train_x @ true_w) + 1)/2
    val_y = (torch.sign(val_x @ true_w) + 1)/2

    return {"train":(x1, x2, y), "val":(val_x, val_y), "true_weights":true_w}

# %%
