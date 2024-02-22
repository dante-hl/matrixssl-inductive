import torch
import numpy as np


def generate_cube_data(n: int = 50000, v: int = 12500, d: int = 50, k: int = 10, labelling: str = 'single'):
    """
    Generates `n` points in the `d`-dimensional hypercube {-1, 1}^d, with a train-test split of n-v:v.

    Training points are sampled uniformly in the d-dimensional hypercube {-1, 1}^d. Two augmentations are produced from
    each point by independently scaling each of the last d-k dimensions by a uniformly sampled constant within (0, 1].
    The first `k` and remaining d-k dimensions correspond to invariant and spurious features respectively.
    Test points are simply sampled uniformly in {-1, 1}^d, with no augmentations.

    The problem type depends on `labelling`. If `labelling` is 'single', then the label is the sign of a single
    dimension chosen randomly from the first k invariant dimensions. If `labelling` is 'all', then the label is the
    sign of some random linear classifier on the invariant dimensions, with weights sampled from a Standard Gaussian.

    Returns train-test data tuple (x1, x2, y), (val_x, val_y)
    Training set: x1, x2, (n-v, d) arrays of augmented data, and labels y (n-v,).
    Validation set: val_x (v, d), val_y (v,)
    """
    assert k < d  # can it be equal..?
    natural = 2 * torch.rand((n, d)) - 1  # uniform in (-1, 1)
    val_x = natural[-v:, :]  # (v, d)
    train_x = natural[:d - v, :]  # (n-v, d)
    tau1, tau2 = torch.ones_like(train_x), torch.ones_like(train_x)
    tau1[:, k:] *= torch.rand((n - v, d - k))
    tau2[:, k:] *= torch.rand((n - v, d - k))
    x1, x2 = train_x * tau1, train_x * tau2
    if labelling == 'single':
        index = np.random.randint(k)
        y = torch.sign(train_x[:, index])
        val_y = torch.sign(val_x[:, index])
    else:
        raise Exception("Invalid input to 'labelling'.")
    return (x1, x2, y), (val_x, val_y)
