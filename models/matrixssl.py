# %%
import torch
import torch.nn as nn
#  %%

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


def uniformity_loss(a1: torch.tensor, a2: torch.tensor):
    """
    Given uncentered (n_ft, b_size) matrices a1, a2, calculate their uniformity loss
    """
    assert a1.shape == a2.shape
    d = a1.shape[0]
    b = a1.shape[1]
    H_b = centering_matrix(b).detach()
    cross_cov = (-1 / b) * a1 @ H_b @ a2.T
    full = ((-1. / d) * torch.eye(d) @ matrix_log(cross_cov)) + cross_cov
    return torch.trace(full)


def alignment_loss(a1: torch.tensor, a2: torch.tensor, gamma: float):
    """
    Given uncentered (n_ft, b_size) matrices a1, a2, calculate their alignment loss
    """
    assert a1.shape == a2.shape
    # d = a1.shape[0]
    b = a1.shape[1]
    H_b = centering_matrix(b).detach()
    cross_cov = (-1. / b) * a1 @ H_b @ a2.T
    autocov1 = (1. / b) * a1 @ H_b @ a1.T
    autocov2 = (1. / b) * a2 @ H_b @ a2.T
    matrix_ce = torch.trace(-autocov1 @ matrix_log(autocov2) + autocov2)
    loss = -torch.trace(cross_cov) + gamma * matrix_ce
    return loss


def mssl_loss(z1: torch.tensor, z2: torch.tensor):
    """
    Given embedding matrices z1, z2, return MatrixSSL loss. This is a naive loss, in the sense that we assume that architecture employed is a symmetric encoder, just acting on two differently augmented sets of images
    """
    u_loss = uniformity_loss(z1, z2)
    a_loss = alignment_loss(z1, z2)
    return u_loss + a_loss, {"uniformity": u_loss, "alignment": a_loss}


def symmetrized_mssl_loss(p1: torch.tensor, p2: torch.tensor, z1: torch.tensor, z2: torch.tensor):
    """
    Alternative version of the MatrixSSL loss, which accounts for the fact that the encoder used in practice is assymetric and utilizes a stop gradient and exponential moving average for one of the networks. Requires as input two (target, predictor) embedding pairs (p1, z2) and (p2, z1), which are the outputs of augmentations x1, x2 passed through target and predictor networks respectively.
    """
    u_loss = uniformity_loss(p1, z2) + uniformity_loss(p2, z1)
    a_loss = alignment_loss(p1, z2) + alignment_loss(p2, z1)
    return u_loss + a_loss, {"uniformity": u_loss, "alignment": a_loss}


class MatrixSSL(nn.module):
    def __init__(self, backbone, emb_dim, gamma: float = 1.0, assym=True):
        """
        backbone: encoder/embedding function
        emb_dim: embedding dimension 
        gamma: TODO figure this out
        assym: whether we use assymetric siamese (online-target) networks, with momentum averaging for the target network
        """
        super().__init__()
        self.assym = assym

        self.gamma = gamma
        self.backbone = backbone
        self.emb_dim = emb_dim
        if assym:
            self.projector = nn.Sequential(nn.Linear(emb_dim, emb_dim, bias=False),
                                           nn.BatchNorm1d(emb_dim),
                                           nn.ReLU(inplace=True), # first layer
                                           nn.Linear(emb_dim, emb_dim, bias=False),
                                           nn.BatchNorm1d(emb_dim),
                                           nn.ReLU(inplace=True), # second layer
                                        #    self.encoder.fc,
                                           nn.BatchNorm1d(emb_dim, affine=False) 
                                          ) # TODO figure if BatchNorm1d needed
            self.predictor = nn.Sequential() # TODO FILL
            self.online = nn.Sequential(self.backbone, self.projector, self.predictor)
            self.target = nn.Sequential(self.backbone, self.projector)
        else:
            self.encoder = nn.Sequential(self.backbone)

    def forward(self, x1, x2):
        if self.assym:
            z1, z2 = self.online(x1), self.online(x2)
            with torch.no_grad():
                p1, p2 = self.target(x1), self.target(x2)
            loss, d_dict = symmetrized_mssl_loss(p1,p2,z1,z2)
            return {'loss': loss, 'd_dict': d_dict}
        else: # worry about this later..
            # f = self.encoder
            # z1 = f(x1)
            # # z1, z2 = f(x1), f(x2)
            # loss, d_dict = mssl_loss(z1, z2)
            # return loss, d_dict
            pass

