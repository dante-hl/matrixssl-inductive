import torch
import numpy as np


def generate_cube_data(n: int = 50000, d: int = 50, k: int = 10, labelling: str = 'single'):
    """
    Generates `n` points in the `d`-dimensional hypercube as follows: n points are sampled uniformly in the
    d-dimensional hypercube {-1, 1}^d. The last d-k dimensions are then independently scaled by a uniformly
    sampled constant within range (0, 1]. The first `k` (<d) and remaining d-k dimensions correspond
    to invariant and spurious features respectively.

    The problem type depends on `labelling`. If `labelling` is 'single', then the label is the sign of a single
    dimension chosen randomly from the first k invariant dimensions. If `labelling` is 'all', then the label is the
    sign of some random linear classifier on the invariant dimensions, with weights sampled from a Standard Gaussian.

    Returns x1, x2, two (n, d) arrays of augmented data, and y, the labels (n,). All have the same order of data.
    """
    assert k < d  # can it be equal..?
    natural = 2 * torch.rand((n, d)) - 1  # uniform in (-1, 1)
    tau1, tau2 = torch.ones(n, d), torch.ones(n, d)
    tau1[:, k:] *= torch.rand((n, d - k))
    tau2[:, k:] *= torch.rand((n, d - k))
    x1, x2 = natural * tau1, natural * tau2
    if labelling == 'single':
        index = np.random.randint(k)
        y = torch.sign(natural[:, index])
    else:
        raise Exception("Invalid input to 'labelling'.")
    return x1, x2, y
