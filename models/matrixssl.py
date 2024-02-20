import torch
import torch.nn as nn


def centering_matrix(b: int):
    """
    Return centering matrix of dimension b
    """
    return torch.eye(b) - (torch.ones((b, b)) / b)


def matrix_log(X: torch.tensor, order: int = 4):
    """
    Calculates matrix logarithm via power series approximation specified by 'order'.
    """
    assert X.shape[0] == X.shape[1]
    d = X.shape[0]
    series = torch.zeros_like(X)
    for i in range(1, order + 1):
        series += torch.linalg.matrix_power(X - torch.eye(d), i) * ((-1) ** (i + 1)) / i
    return series


def alt_matrix_log(X: torch.tensor, order: int = 4):
    """
    Alternate version of matrix_log, perhaps computationally lighter?
    """
    assert X.shape[0] == X.shape[1]
    d = X.shape[0]
    series = torch.zeros_like(X)
    mult = X - torch.eye(d)
    summand = mult
    for i in range(1, order + 1):
        if i % 2 == 1:
            series = series + (1 / i) * summand
        else:
            series = series - (1 / i) * summand
        summand = summand @ mult
    return series


def uniformity_loss(z1: torch.tensor, z2: torch.tensor):
    """
    Given uncentered (n_ft, b_size) embedding matrices z1, z2, calculate their uniformity loss
    """
    assert z1.shape == z2.shape
    d = z1.shape[0]
    b = z1.shape[1]
    H_b = centering_matrix(b).detach()
    cross_cov = (-1 / b) * z1 @ H_b @ z2.T
    full = ((-1. / d) * torch.eye(d) @ matrix_log(cross_cov)) + cross_cov
    return torch.trace(full)


# TODO: Look into what things have to be changed because of autograd/for efficiency..


def alignment_loss(z1: torch.tensor, z2: torch.tensor, gamma: float):
    """
    Given uncentered (n_ft, b_size) embedding matrices z1, z2, calculate their alignment loss
    """
    assert z1.shape == z2.shape
    # d = z1.shape[0]
    b = z1.shape[1]
    H_b = centering_matrix(b).detach()
    cross_cov = (-1. / b) * z1 @ H_b @ z2.T
    autocov1 = (1. / b) * z1 @ H_b @ z1.T
    autocov2 = (1. / b) * z2 @ H_b @ z2.T
    matrix_ce = torch.trace(-autocov1 @ matrix_log(autocov2) + autocov2)
    loss = -torch.trace(cross_cov) + gamma * matrix_ce
    return loss


def matrix_ssl_loss(z1: torch.tensor, z2: torch.tensor):
    """
    Given uncentered embedding matrices z1, z2, calculate the total Matrix SSL loss
    """
    u_loss = uniformity_loss(z1, z2)
    a_loss = alignment_loss(z1, z2)
    return u_loss + a_loss, {"uniformity": u_loss, "alignment": a_loss}


class MatrixSSL(nn.module):
    def __init__(self, backbone, gamma: float = 1.0):
        super().__init__()

        self.gamma = gamma
        self.backbone = backbone
        self.encoder = nn.Sequential(self.backbone)

    def forward(self, x1, x2):
        f = self.encoder
        z1, z2 = f(x1), f(x2)
        loss, d_dict = matrix_ssl_loss(z1, z2)
        return loss, d_dict
