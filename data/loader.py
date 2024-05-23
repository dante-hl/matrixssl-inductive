# %%
import torch
import numpy as np

# %%
# ignoring labelling: str = 'single', for now, add more if you need
def generate_cube_data(
    n: int = (2 ** 16) + 12500, 
    v: int = 12500,
    d: int = 50, 
    k: int = 10,
    nat='bool',
    aug='mult', 
    label=None, 
    tau_max=None):
    """
    Generates `n` points in the `d`-dimensional hypercube {-1, 1}^d, with a train-val split of n-v:v.

    Natural data are sampled according to scheme specified by `nat`. Augmentations (and labels) are then produced by the scheme specified by `aug`.

    `n`: total number of data (train + val)
    `v`: number of validation points
    `d`: total dimension of each datum
    `k`: number of invariant features

    `nat`: string specifying natural data sampling scheme. can be one of {'bool', 'unif'}. if 'bool', then each element of the natural data is either -1 or 1, sampled with equal probability. If 'unif', then natural data is sampled uniformly within the hypercube [-1, 1]^d

    `aug`: string specifying augmentation scheme:
        'mult': scales each invariant dimension independently by a scalar tau ~ U(0, 1]. resulting labeling function  uses a linear classifier with randomly sampled weights from N(0, 1) on the first k (invariant) dimensions of data
        'add': for each pair of dimensions, samples tau ~ U(0, 1], then adds to one dimension and subtracts from the other. requires an even number of spurious dimensions to guarantee label preservation. labeling function returns the sign of the sum of all spurious dimension values.
        'corr': requires number of augmented dimensions to be even. given natural data 
        (x_{:k}, x_{k:k+h}, x_{k+h:k+2h}), samples tau ~ U[-1, 1]^h, and returns augmentation 
        (x_{:k}, x_{k:k+h} + tau x_{k+h:k+2h}, x_{k+h:k+2h} + tau x_{k:k+h}).
    
    `label`: string specifying how labels are generated:
        'weights' uses a weighted linear combination of the k unagumentedion dimensions
        'inv1d' means y = sign(first unaugmented dimension)
        'invkd' means y = sign(sum of all unagumented dimensions)
        'spur' means y = sign(sum of all pairs of augmented dimensions)
        (sp1d does't really mean anything intuitively, so not inclued)

    Returns dictionary: {"train":(x1, x2, y), "val":(val_x, val_y), "true_w":true_w}
    Training set: x1, x2, (n-v, d) arrays of augmented data, and labels y (n-v,).
    Validation set: val_x (v, d), val_y (v,)
    true_w: if aug='mult', then the true linear classif weights. None otherwise
    """
    assert k < d 
    if nat == 'bool':
        natural = ((2 * torch.randint(low=0, high=2, size=(n, d))) - 1).float() # boolean in {-1, 1}
    elif nat == 'unif':
        natural = (2 * torch.rand((n, d))) - 1  # uniform in (-1, 1)
    else:
        raise Exception("Invalid natural sampling scheme, expected one of {'bool', 'unif'}")
    
    val_x = natural[-v:, :]  # (v, d)
    train_x = natural[:n - v, :]  # (n-v, d)
    # print(f'Train x {train_x}')

    true_w = None
    if aug == 'mult':
        # augment by scaling down invariant dimensions
        tau1, tau2 = torch.ones_like(train_x), torch.ones_like(train_x)
        tau1[:, k:] *= torch.rand((n - v, d - k))
        tau2[:, k:] *= torch.rand((n - v, d - k))
        x1, x2 = train_x * tau1, train_x * tau2
    elif aug == 'add':
        # augment by adding and subtracting tau
        assert (d - k) % 2 == 0
        half = int((d-k)/2) # half number of spurious features
        additive1, additive2 = torch.zeros_like(train_x), torch.zeros_like(train_x)

        tau1 = torch.rand((n-v, half))
        additive1[:, k:k + half] += tau1
        additive1[:, k+half:] -= tau1

        tau2 = torch.rand((n-v, half))
        additive2[:, k:k+half] += tau2
        additive2[:, k+half:] -= tau2

        x1, x2 = train_x + additive1, train_x + additive2
        # originally used spur labeling
    elif aug == 'corr':
        assert tau_max > 0
        tau_l = - tau_max
        tau_u = tau_max
        # SAME TAU FOR ALL DIMENSIONS (and all data points as well)
        # half = int((d-k)/2) # half number of spurious features
        # additive1, additive2 = torch.zeros_like(train_x), torch.zeros_like(train_x)

        # tau1 = tau_l + torch.rand(1) * (tau_u - tau_l)
        # additive1[:, k:k + half] += tau1 * train_x[:, k+half:]
        # additive1[:, k+half:] += tau1 * train_x[:, k:k + half]

        # tau2 = tau_l + torch.rand(1) * (tau_u - tau_l)
        # additive2[:, k:k + half] += tau2 * train_x[:, k+half:]
        # additive2[:, k+half:] += tau2 * train_x[:, k:k + half]

        # x1, x2 = train_x + additive1, train_x + additive2

        # DIFFERENT TAU FOR EACH DIMENSION (and for each data point as well)
        half = int((d-k)/2) # half number of spurious features
        additive1, additive2 = torch.zeros_like(train_x), torch.zeros_like(train_x)

        tau1 = tau_l + torch.rand((n-v), half) * (tau_u - tau_l)
        additive1[:, k:k + half] += tau1 * train_x[:, k+half:]
        additive1[:, k+half:] += tau1 * train_x[:, k:k + half]

        tau2 = tau_l + torch.rand((n-v), half) * (tau_u - tau_l)
        additive2[:, k:k + half] += tau2 * train_x[:, k+half:]
        additive2[:, k+half:] += tau2 * train_x[:, k:k + half]

        x1, x2 = train_x + additive1, train_x + additive2
        # can use inv1d, invkd labeling, spur labeling only if embedding dimensions is size 
        # inv1d, invkd: any range
        # spur: range must be chosen to keep sign of sum positive
    else:
        raise Exception("Invalid augmentation, expected one of {'mult', 'add'}")

    if label == 'weights':
        # instantiate labeling function weights (true_w)
        true_w = torch.zeros(d)
        true_w[:k] = torch.randn(k)
        # create labels, change from {-1, 1} to {0, 1} labels - for linear classification loss
        y = (torch.sign(train_x @ true_w) + 1)/2
        val_y = (torch.sign(val_x @ true_w) + 1)/2
    # TODO NEED TO ACCOUNT FOR POSSIBILITY OF BOOLEAN SUM BEING 0. MADE CHANGES BELOW.
    # ORIGINAL STILL INCLUDED AND COMMENTED OUT
    elif label == 'inv1d': # label depends on first unaugmented dimension
        y = (torch.sign(train_x[:, 0]) + 1)/2
        val_y = (torch.sign(val_x[:, 0]) + 1)/2
    elif label == 'invkd': # label depends on k unaugmented dimensions
        sumofk = torch.sum(train_x[:, :k], dim=1)
        sumofk_val = torch.sum(val_x[:, :k], dim=1)
        sumofk[sumofk == 0] = 1 # if sum is 0, set label to 1
        sumofk_val[sumofk_val == 0] = 1
        y = (torch.sign(sumofk) + 1) / 2
        val_y = (torch.sign(sumofk_val) + 1) / 2

        # y = (torch.sign(torch.sum(train_x[:, :k], dim=1)) + 1)/2
        # val_y = (torch.sign(torch.sum(val_x[:, :k], dim=1)) + 1)/2
    elif label == 'spur': # label depends on sum of all augmented dimensions
        sumofspur = torch.sum(train_x[:, k:], dim=1)
        sumofspur_val = torch.sum(val_x[:, k:], dim=1)
        sumofspur[sumofspur == 0] = 1
        sumofspur_val[sumofspur_val == 0] = 1
        y = (torch.sign(sumofspur) + 1) / 2
        val_y = (torch.sign(sumofspur_val) + 1) / 2
        # y = (torch.sign(torch.sum(train_x[:, k:], dim=1)) + 1)/2
        # val_y = (torch.sign(torch.sum(val_x[:, k:], dim=1)) + 1)/2
        
    return {"train":(x1, x2, y), "val":(val_x, val_y), "true_w":true_w}


def generate_correlated_normal_data(n: int = (2 ** 16) + 12500, v: int = 12500, d: int = 25, K:int=33):
    """
    d: dimension of each feature
    K: number of features

    alpha chosen uniformly from [0, 1]^K
    ground truth feature vectors chosen uniformly in R^d
    different features are sampled independently
    different elements in each feature are samepled iid and normally, but the features across
    """
    alpha = torch.rand(d)
    gt = torch.rand(K, d) # ground truth feature vectors (rows of array)

#     pass

# %%
