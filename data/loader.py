# %%
import torch
import numpy as np

# %%
# ignoring labelling: str = 'single', for now, add more if you need
def generate_cube_data(n: int = (2 ** 16) + 12500, v: int = 12500, d: int = 50, k: int = 10, augmentation='mult'):
    """
    Generates `n` points in the `d`-dimensional hypercube {-1, 1}^d, with a train-val split of n-v:v.

    Training points are sampled uniformly in the d-dimensional hypercube {-1, 1}^d. Augmentations are produced from these sampled points by independently scaling each of the last d-k dimensions by a uniformly sampled constant within (0, 1]. The first `k` and remaining d-k dimensions correspond to invariant and spurious features respectively.Validation points are sampled uniformly in {-1, 1}^d, with no augmentations.

    n: total number of data (train + val)
    v: number of validation points
    d: total dimension of each datum
    k: number of invariant features

    augmentation: string specifying type of augmentation: note that this also specifies the labeling function, since augmentations must be label preserving.
        'mult': scales each invariant dimension independently by a scalar tau ~ U(0, 1]. resulting labeling function  uses a linear classifier with randomly sampled weights from N(0, 1) on the first k (invariant) dimensions of data
        'add': for each pair of dimensions, samples tau ~ U(0, 1], then adds to one dimension and subtracts from the other. requires an even number of spurious dimensions to guarantee label preservation. labeling function returns the sign of the sum of all spurious dimension values.

    Returns dictionary: {"train":(x1, x2, y), "val":(val_x, val_y), "truth":truth}
    Training set: x1, x2, (n-v, d) arrays of augmented data, and labels y (n-v,).
    Validation set: val_x (v, d), val_y (v,)
    Truth: if augmentation='mult', then the true linear classif weights. if 'add', then the string 'spurious_sum' (to indicate summing spurious dimensions)
    """
    assert k < d  # can it be equal..?
    natural = 2 * torch.rand((n, d)) - 1  # uniform in (-1, 1)
    val_x = natural[-v:, :]  # (v, d)
    train_x = natural[:n - v, :]  # (n-v, d)
    # print(f'Train x {train_x}')

    if augmentation == 'mult':
        # augment by scaling down invariant dimensions
        tau1, tau2 = torch.ones_like(train_x), torch.ones_like(train_x)
        tau1[:, k:] *= torch.rand((n - v, d - k))
        tau2[:, k:] *= torch.rand((n - v, d - k))
        x1, x2 = train_x * tau1, train_x * tau2
        # instantiate labeling function weights (truth)
        truth = torch.zeros(d)
        truth[:k] = torch.randn(k)
        # create labels, change from {-1, 1} to {0, 1} labels - for linear classification loss
        y = (torch.sign(train_x @ truth) + 1)/2
        val_y = (torch.sign(val_x @ truth) + 1)/2
    elif augmentation == 'add':
        # augment by adding and subtracting tau
        assert (d - k) % 2 == 0
        half = int((d-k)/2) # half number of spurious features
        tau1, tau2 = torch.zeros_like(train_x), torch.zeros_like(train_x)

        additive1 = torch.rand((n-v, half))
        tau1[:, k:k + half] += additive1
        tau1[:, k+half:] -= additive1

        additive2 = torch.rand((n-v, half))
        tau2[:, k:k + half] += additive2
        tau2[:, k+half:] -= additive2

        # print(f'Additive1: {additive1}')
        # print(f'Additive2: {additive2}')

        x1, x2 = train_x + tau1, train_x + tau2
        # create labels: {-1, 1} -> {0, 1}
        y = (torch.sign(torch.sum(train_x[:, k:], dim=1)) + 1)/2
        val_y = (torch.sign(torch.sum(val_x[:, k:], dim=1)) + 1)/2

        truth = 'spurious_sum'
    else:
        raise Exception("Invalid augmentation, expected one of {'mult', 'add'}")
    # print(f"weights: {true_w}")

    return {"train":(x1, x2, y), "val":(val_x, val_y), "truth":truth}

# %%
