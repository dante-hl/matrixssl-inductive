# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
#  %%

def centering_matrix(b: int):
    """
    Return centering matrix of dimension b
    """
    return torch.eye(b) - (torch.ones((b, b)) / b)


# def matrix_log(X: torch.tensor, order: int = 4):
#     """
#     Calculates matrix logarithm via power series approximation specified by 'order'.
#     """
#     assert X.shape[0] == X.shape[1]
#     d = X.shape[0]
#     series = torch.zeros_like(X)
#     for i in range(1, order + 1):
#         series += torch.linalg.matrix_power(X - torch.eye(d), i) * ((-1) ** (i + 1)) / i
#     return series


def matrix_log(X: torch.tensor, order: int = 4):
    """
    matrix_log, from MEC. uses taylor expansion approximation
    """
    assert X.shape[0] == X.shape[1]
    d = X.shape[0]
    series = torch.zeros_like(X).detach()
    mult = X - torch.eye(d)
    summand = mult
    for i in range(1, order + 1):
        if i % 2 == 1:
            series = series + (1. / float(i)) * summand
        else:
            series = series - (1. / float(i)) * summand
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
    cross_cov = (1. / b) * a1 @ H_b @ a2.T
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
    cross_cov = (1. / b) * a1 @ H_b @ a2.T
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


def symmetrized_mssl_loss(p1: torch.tensor, p2: torch.tensor, z1: torch.tensor, z2: torch.tensor, gamma: float):
    """
    Alternative version of the MatrixSSL loss, which accounts for the fact that the encoder used in practice is asymmetric and utilizes a stop gradient and exponential moving average for one of the networks. Requires as input two (target, predictor) embedding pairs (p1, z2) and (p2, z1), which are the outputs of augmentations x1, x2 passed through target and predictor networks respectively.
    """
    # I think you just have to normalize inputs before centering...
    u_loss = uniformity_loss(p1, z2) + uniformity_loss(p2, z1)
    # alignment loss returns huge value
    a_loss = alignment_loss(p1, z2, gamma) + alignment_loss(p2, z1, gamma)
    return u_loss + a_loss, {"uniformity": u_loss, "alignment": a_loss}



def mce_loss_func(p, z, lamda=1., mu=1., order=4, align_gamma=0.003, correlation=True, logE=False):
    """
    Taken from MCE
    """
    p = F.normalize(p)  # wait... need to be normalized...???
    z = F.normalize(z)

    m = z.shape[0]
    n = z.shape[1]
    # print(m, n)
    J_m = centering_matrix(m).detach()
    
    if correlation:
        P = lamda * torch.eye(n)
        Q = (1. / m) * (p.T @ J_m @ z) + mu * torch.eye(n)
        # return torch.trace(- P @ matrix_log(Q, order))
    else:
        P = (1. / m) * (p.T @ J_m @ p) + mu * torch.eye(n)
        Q = (1. / m) * (z.T @ J_m @ z) + mu * torch.eye(n)
        # return torch.trace(- P @ matrix_log(Q, order) + Q)
    



class MatrixSSL(nn.Module):
    def __init__(self, backbone, emb_dim, hidden_dim, gamma: float = 1.0, asym=True, momentum=0.9):
        """
        backbone: encoder/embedding function
        emb_dim: embedding dimension 
        hidden_dim: hidden dimension in predictor network
        gamma: weight ratio in alignment loss (see alignment loss func)
        asym: whether we use asymmetric siamese (online-target) networks, with momentum averaging for the target network
        momentum: momentum parameter for moving average of target network. only required if asym = True. must be within [0, 1]
        """
        super().__init__()
        self.asym = asym

        self.gamma = gamma
        self.backbone = backbone
        self.emb_dim = emb_dim
        if asym:
            self.momentum = momentum
            self.hidden_dim = hidden_dim
            # start with making everything linear, just one layer
            # make each of proejctor and predictor smaller....
            self.projector = nn.Sequential(nn.Linear(emb_dim, emb_dim, bias=False),
                                           nn.BatchNorm1d(emb_dim),
                                           nn.ReLU(inplace=True), # first layer
                                           nn.Linear(emb_dim, emb_dim, bias=False),
                                           nn.BatchNorm1d(emb_dim),
                                           nn.ReLU(inplace=True), # second layer
                                        #    self.encoder.fc,
                                           nn.BatchNorm1d(emb_dim, affine=False) 
                                          ) # TODO figure if BatchNorm1d needed
            self.predictor = nn.Sequential(nn.Linear(emb_dim, hidden_dim, bias=False),
                                        nn.BatchNorm1d(hidden_dim),
                                        nn.ReLU(inplace=True), # hidden layer
                                        nn.Linear(hidden_dim, emb_dim)) # output layer
            self.online = nn.Sequential(self.backbone, self.projector, self.predictor)
            self.target = nn.Sequential(self.backbone, self.projector)
        else:
            self.encoder = nn.Sequential(self.backbone)

    def forward(self, x1, x2):
        if self.asym:
            z1, z2 = self.online(x1), self.online(x2)
            with torch.no_grad():
                p1, p2 = self.target(x1), self.target(x2)
            # loss, d_dict = symmetrized_mssl_loss(p1,p2,z1,z2, self.gamma)
            loss = (
                    mce_loss_func(p2.T, z1.T, correlation=True)
                    + mce_loss_func(p1.T, z2.T, correlation=True)
                    + self.gamma * mce_loss_func(p2.T, z1.T, correlation=False)
                    + self.gamma * mce_loss_func(p1.T, z2.T, correlation=False)
                    ) * 0.5
            return {'loss': loss, 'd_dict': []} #d_dict}
        else: # worry about this later..
            # f = self.encoder
            # z1 = f(x1)
            # # z1, z2 = f(x1), f(x2)
            # loss, d_dict = mssl_loss(z1, z2)
            # return loss, d_dict
            pass

