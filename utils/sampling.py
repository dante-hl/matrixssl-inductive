# %%
import torch
# %%
# basic example: d = 3, k = 1
def generate_aug_from_bool_aug(x, n, k):
    """
    given a single input augmentation x, shape (d,), generated from boolean natural data {-1, 1}^d and a correlation augmentation scheme on the last d-k dimensions, sample n other possible augmentations that could form a positive pair with x, shape (n, d)

    works for augmentations with tau sampled from [-a, a] for any a in [0, 1]
    """
    d = x.shape[0]
    assert (d - k) % 2 == 0
    h = int((d-k)/2)
    natural = torch.zeros(d)
    natural[:k] = x[:k]
    for i in range(h):
        c_ki, c_khi = x[k+i], x[k+h+i]
        if c_ki >= 0 and c_ki <= 2:
            if c_khi >= 0 and c_khi <= 2:
                # c_ki, c_khi in [0, 2]
                natural[k+i] = 1
                natural[k+h+i] = 1
            elif c_khi >= -2 and c_khi <= 0:
                # c_ki in [0, 2], c_khi in [-2, 0]
                natural[k+i] = 1
                natural[k+h+i] = -1
            else:
                raise Exception('invalid values for input x!!')
        elif c_ki >= -2 and c_ki <= 0:
            if c_khi >= 0 and c_khi <= 2:
                # c_ki in [-2, 0], c_khi in [0, 2]
                natural[k+i] = -1
                natural[k+h+i] = 1
            elif c_khi >= -2 and c_khi <= 0:
                # c_ki, c_khi in [-2, 0]
                natural[k+i] = -1
                natural[k+h+i] = -1
            else:
                raise Exception('invalid values for input x!!')
        else:
            raise Exception('invalid values for input x!!')

    augments = torch.zeros(n, d)
    augments[:, :k] = natural[:k]

    tau = torch.rand(n, h)
    augments[:, k:k+h] += natural[k:k+h] + (tau * natural[k+h:])
    augments[:, k+h:] += natural[k+h:] + (tau * natural[k:k+h])
    return augments