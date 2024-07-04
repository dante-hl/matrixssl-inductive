# %%
import torch
import numpy as np
from torch.distributions.multivariate_normal import MultivariateNormal

# %%
def train_split(tuple_of_data, train_size):
    """
    splits all data in tuple_of_data into a training and holdout set, with training set size specified by train_size. assumes all data in tuple_of_data have same length in 0th dimension

    train_size can either be an integer (exact number of data in train split) or a float (fraction of data in train split, with actual split rounded down to nearest integer)

    TODO: use case appears to only be for correlated normal data. either make a note on this, or delete this func.
    """
    idx_type = type(train_size)
    # ASSERTIONS
    assert idx_type == int or idx_type == float
    if idx_type == float:
        assert train_size >=0 and train_size <= 1
    n = tuple_of_data[0].shape[0] # length of dim 0 of each data
    if idx_type == int:
        assert train_size >= 0 and train_size <= n

    split_idx = train_size if idx_type == int else int(train_size * n)
    train_list = []
    holdout_list = []
    for data in tuple_of_data:
        train_list.append(data[:split_idx])
        holdout_list.append(data[split_idx:])
    train_tuple, holdout_tuple = tuple(train_list), tuple(holdout_list)
    return train_tuple, holdout_tuple


def label_data(data_matrices, k, labeling):
    """
    Labels each matrix of data in `data_matrices` with labeling scheme given by `labeling`. Assumes data points correspond to rows of matrices
    
    `data_matrices`: list/tuple/iterable of data matrices
    `k`: specifies border for relevant dimensions, depending on the exact augmentation scheme 
    `labeling`: string specifying how labels are generated:
        'weightedk' uses a weighted linear combination of the first k dimensions
        'sum1' means y = sign(first dimension)
        'sumk' means y = sign(sum of first k dimensions)
        'sumd-k' means y = sign(sum of last d-k dimensions), where d dimension of data

    returns: list whose ith term corresponds to the labels of the ith data matrix in `data_matrices`
    """
    d = data_matrices[0].shape[0]
    label_list = [] # list of label vectors corresp to each data matrix in data_tuple
    for data in data_matrices:
        if labeling == 'weightedk':
            # instantiate labeling function weights (true_w)
            true_w = torch.zeros(d)
            true_w[:k] = torch.randn(k)
            # create labels, change from {-1, 1} to {0, 1} labels - for linear classification loss
            y = (torch.sign(data @ true_w) + 1)/2
        elif labeling == 'sum1': # label depends on first dimension
            y = (torch.sign(data[:, 0]) + 1)/2
        elif labeling == 'sumk': # label depends on first k dimensions
            sum_of_k = torch.sum(data[:, :k], dim=1)
            sum_of_k[sum_of_k == 0] = 1 # if sum is 0, set label to 1
            y = (torch.sign(sum_of_k) + 1) / 2
        elif labeling == 'sumd-k': # label depends on sum of all dimensions after kth dim
            sum_after_k = torch.sum(data[:, k:], dim=1)
            sum_after_k[sum_after_k == 0] = 1
            y = (torch.sign(sum_after_k) + 1) / 2
        label_list.append(y)
    return label_list


def generate_cube_augs(
    n: int = (2 ** 16) + 12500, 
    d: int = 50, 
    k: int = 10,
    nat='bool',
    aug='mult', 
    tau_max=None):
    """
    Generates `n` augmentation pairs in the `d`-dimensional hypercube {-1, 1}^d, using a natural data sampling scheme, followed by an augmentation scheme.

    Natural data are sampled according to `nat`. Augmentation scheme specified by `aug`.

    `n`: total number of data (train + val)
    `v`: number of validation points
    `d`: total dimension of each datum
    `k`: number of unaugmented/augmented features

    `nat`: string specifying natural data sampling scheme. one of {'bool', 'unif'}. 
        if 'bool', then each element of natural data is -1 or 1, sampled with equal probability
        if 'unif', then natural data is sampled uniformly within the hypercube [-1, 1]^d

    `aug`: string specifying augmentation scheme. one of {'mult', 'add', 'corr'}
        'mult': scales last d-k dimensions independently by a scalar tau ~ U(0, 1].
        'add': for each pair of dimensions, samples tau ~ U(0, 1], then adds to one dimension and subtracts from the other. requires an even number of spurious dimensions to guarantee label preservation.
        'corr': requires number of augmented dimensions to be even. given natural data 
        (x_{:k}, x_{k:k+h}, x_{k+h:k+2h}), samples tau ~ U[-1, 1]^h, and returns augmentation 
        (x_{:k}, x_{k:k+h} + tau x_{k+h:k+2h}, x_{k+h:k+2h} + tau x_{k:k+h}).

    Returns tuple (x1, x2) of augmentation matrices (one per row). augmentation pairs correspond to the same index across x1 and x2
    """
    assert k < d 
    if nat == 'bool':
        natural = ((2 * torch.randint(low=0, high=2, size=(n, d))) - 1).float() # (n, d), boolean in {-1, 1}
    elif nat == 'unif':
        natural = (2 * torch.rand((n, d))) - 1  # (n, d), uniform in (-1, 1)
    else:
        raise Exception("Invalid natural sampling scheme, expected one of {'bool', 'unif'}")

    if aug == 'mult':
        # augment by scaling down last d-k dimensions
        tau1, tau2 = torch.ones_like(natural), torch.ones_like(natural) # (n, d)
        tau1[:, k:] *= torch.rand((n, d - k))
        tau2[:, k:] *= torch.rand((n, d - k))
        x1, x2 = natural * tau1, natural * tau2
    elif aug == 'add':
        # augment by adding and subtracting tau
        assert (d - k) % 2 == 0
        half = int((d-k)/2) # half number of spurious features
        toadd1, toadd2 = torch.zeros_like(natural), torch.zeros_like(natural)

        tau1 = torch.rand((n, half))
        toadd1[:, k:k+half] += tau1
        toadd1[:, k+half:] -= tau1

        tau2 = torch.rand((n, half))
        toadd2[:, k:k+half] += tau2
        toadd2[:, k+half:] -= tau2

        x1, x2 = natural + toadd1, natural + toadd2
        # originally used spur labeling
    elif aug == 'corr':
        assert tau_max > 0
        tau_l = - tau_max
        tau_u = tau_max
        # SAME TAU FOR ALL DIMENSIONS (and all data points as well)
        # half = int((d-k)/2) # half number of spurious features
        # toadd1, toadd2 = torch.zeros_like(natural), torch.zeros_like(natural)

        # tau1 = tau_l + torch.rand(1) * (tau_u - tau_l) # (1,) ~ U[tau_l, tau_u]
        # toadd1[:, k:k + half] += tau1 * natural[:, k+half:]
        # toadd1[:, k+half:] += tau1 * natural[:, k:k + half]

        # tau2 = tau_l + torch.rand(1) * (tau_u - tau_l)
        # toadd2[:, k:k + half] += tau2 * natural[:, k+half:]
        # toadd2[:, k+half:] += tau2 * natural[:, k:k + half]

        # x1, x2 = natural + toadd1, natural + toadd2

        # DIFFERENT TAU FOR EACH DIMENSION (and for each data point as well)
        half = int((d-k)/2) # half number of spurious features
        toadd1, toadd2 = torch.zeros_like(natural), torch.zeros_like(natural)

        tau1 = tau_l + torch.rand(n, half) * (tau_u - tau_l) # (n, half) ~iid U[tau_l, tau_u]
        toadd1[:, k:k+half] += tau1 * natural[:, k+half:]
        toadd1[:, k+half:] += tau1 * natural[:, k:k+half]

        tau2 = tau_l + torch.rand(n, half) * (tau_u - tau_l)
        toadd2[:, k:k+half] += tau2 * natural[:, k+half:]
        toadd2[:, k+half:] += tau2 * natural[:, k:k+half]

        x1, x2 = natural + toadd1, natural + toadd2
        # can use inv1d, invkd labeling, spur labeling only if embedding dimensions is size 
        # inv1d, invkd: any range
        # spur: range must be chosen to keep sign of sum positive
    else:
        raise Exception("Invalid augmentation, expected one of {'mult', 'add', 'corr'}")
    return {"train": (x1, x2),}


# currently: modifying so this function can take in alphas and gt_vecs /gt_covs to generate new val and test data
def generate_correlated_normal_augs(n: int = (2 ** 16), num_feats:int=5, feat_dim: int = 5, gt_dict=None):
    """
    Generates `n` pairs of augmentations, with each augmentation a concatenation of `num_feats` features of dimension `feat_dim`. Each feature in each augmentation is distributed according to the standard (multivar) normal distribution, but the covariance between the same feature across the two augmentations is given by a rank one matrix alpha vv.T, where v is some randomly sampled (`feat_dim`,) ground truth vector and alpha is some randomly sampled constant ~ U[0, 1]. Ground truth may also be provided via a dictionary `gt_dict`. See details below.

    num_feats: number of features
    feat_dim: dimension of each feature
    gt_dict: dictionary that must contain "alphas" : list of alpha constants, and must contain at least one of 
    "gt_vecs" : list of gt vectors, OR "gt_covs" : list of joint augmentation covariances

    returns
    (x1, x2) datasets corresponding to augmentation pairs (data index wise)
    alphas: list of constants multiplying each covariance
    gt_vecs: list of ground truth feature vectors along which (parts of) augment pairs are +vely correlated
    """
    if gt_dict: # if ground-truth dictionary provided
        assert isinstance(gt_dict, dict), "gt_dict must be a dictionary"
        alphas = gt_dict["alphas"]
        gt_vecs = getattr(gt_dict, "gt_vecs", None)
        gt_covs = getattr(gt_dict, "gt_covs", None)
        assert gt_vecs or gt_covs, "gt_dict must contain gt_vecs or gt_covs attribute"
        if not gt_covs: # only gt_vecs provided; generate gt_covs from alphas and gt_vecs
            gt_covs = []
            feat_dim = gt_vecs[0].shape[0]
            for alpha, vec in zip(alphas, gt_vecs):
                off_diag = alpha * torch.outer(vec, vec)
                cov = torch.eye(2*feat_dim)
                cov[feat_dim:, :feat_dim] = off_diag
                cov[:feat_dim, feat_dim:] = off_diag
                gt_covs.append(cov)
    else: # if gt dictionary not provided, generate gt yourself
        alphas = []
        gt_vecs = []
        gt_covs = [] # covariance matrices for joint dist of augmentation pairs (u, v)
        for i in range(num_feats):
            while True:
                const = torch.rand(1)
                # const = 1 ####
                vec = torch.randn(feat_dim)
                # vec = vec / torch.norm(vec) ####
                if torch.abs(const * (torch.norm(vec) ** 2)) < 1: # covariance of joint distr of augs is valid (PSD)
                    off_diag = const * torch.outer(vec, vec)
                    cov = torch.eye(2*feat_dim)
                    cov[feat_dim:, :feat_dim] = off_diag
                    cov[:feat_dim, feat_dim:] = off_diag
                    alphas.append(const)
                    gt_vecs.append(vec)
                    gt_covs.append(cov)
                    break

    joint_normals = [MultivariateNormal(loc=torch.zeros(2*feat_dim), covariance_matrix=cov) for cov in gt_covs] # num_feats normals
    samples = [normal.sample((n,)) for normal in joint_normals]
    x1_list = [x[:, :feat_dim] for x in samples] # divide by augmentation group
    x2_list = [x[:, feat_dim:] for x in samples]
    x1 = torch.hstack(x1_list) # concat/stack 'num_feats' features to get a single datum
    x2 = torch.hstack(x2_list)
    return {"train":(x1, x2), "alphas":alphas, "gt_vecs":gt_vecs, "gt_covs":gt_covs}

# %%
