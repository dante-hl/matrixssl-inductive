# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
#  %%

def centering_matrix(b: int):
    """
    Return centering matrix of dimension b
    """
    return torch.eye(b) - (torch.ones((b, b)) / b)


def matrix_log(X: torch.tensor, order: int = 4):
    """
    matrix_log, from MEC. uses taylor expansion approximation
    """
    device = X.device
    assert X.shape[0] == X.shape[1]
    d = X.shape[0]
    series = torch.zeros_like(X).detach().to(device)
    I_d = torch.eye(d).to(device)
    mult = X - I_d
    summand = mult
    for i in range(1, order + 1):
        if i % 2 == 1:
            series = series + (1. / float(i)) * summand
        else:
            series = series - (1. / float(i)) * summand
        summand = summand @ mult
    del mult, I_d
    return series


def mce_loss_func(p, z, lamda=1., mu=1., order=4, align_gamma=0.003, correlation=True, logE=False):
    """
    Taken from MCE
    """
    device = p.device
    p = F.normalize(p)  # wait... need to be normalized...???
    z = F.normalize(z)

    m = z.shape[0]
    n = z.shape[1]
    # print(m, n)
    J_m = centering_matrix(m).detach().to(device)
    mu_I = mu * torch.eye(n).to(device)
    
    if correlation:
        P = lamda * torch.eye(n).to(device)
        Q = (1. / m) * (p.T @ J_m @ z) + mu_I
    else:
        P = (1. / m) * (p.T @ J_m @ p) + mu_I
        Q = (1. / m) * (z.T @ J_m @ z) + mu_I
    return torch.trace(- P @ matrix_log(Q, order))

def mean_norm_diff_loss(z1, z2):
    """
    Calculate alignment loss given by E_{x, x' ~ p+}[||f(x)f(x)^T - f(x')f(x')^T||_F^2]
    i.e the mean of Frobenius norm of difference between outer products). Note this is different from
    the Frobenius norm of the mean difference between outer products (see norm_mean_diff_loss).

    z1, z2: batch of B embeddings of shape (B, d) (???)
    """
    device = z1.device
    d = z1.shape[1]
    J_d = centering_matrix(d).detach().to(device)
    z1 = z1 @ J_d
    z2 = z2 @ J_d
    z1_outer = z1.unsqueeze(2) @ z1.unsqueeze(1) # unsqueeze and batch multiply for outer prods
    z2_outer = z2.unsqueeze(2) @ z2.unsqueeze(1) # (B, d, 1) @ (B, 1, d) -> (B, d, d)
    batched_outer_diff = z1_outer - z2_outer # shape (B, d, d)
    # frobenius norm on last 2 dims
    squared_norms = torch.linalg.matrix_norm(batched_outer_diff, ord='fro', dim=(1, 2)) ** 2
    return torch.mean(squared_norms)

def norm_mean_diff_loss(z1, z2):
    """
    Calculate alignment loss given by || E_{x, x' ~ p+}[f(x)f(x)^T - f(x')f(x')^T] ||_F^2
    i.e the Frobenius norm of the mean difference between outer products. Note this is different from
    the mean of Frobenius norm of difference between outer products (see mean_norm_diff_loss).

    We expect this to not work, as for large batch sizes the term inside the expectation would just approach 0,
    and wouldn't explicit encourage any specific feature learning.
    """
    B = z1.shape[0]
    # d = z1.shape[1]
    return torch.linalg.matrix_norm((z1.T @ z1 - z2.T @ z2)/B) ** 2

def uniformity_loss(*args):
    """
    Calculate uniformity loss. Expects at least one of z1 or z2. If both are provided, uses both.
    """
    if len(args) == 1:
        norms = torch.linalg.vector_norm(args[0], dim=1) ** 2
        return -2 * torch.mean(norms)
    elif len(args) == 2:
        z1_norms = torch.linalg.vector_norm(args[0], dim=1) ** 2
        z2_norms = torch.linalg.vector_norm(args[1], dim=1) ** 2
        return -2 * torch.sqrt(torch.mean(z1_norms) * torch.mean(z2_norms))
    else:
        raise ValueError("Expected 1 or 2 arguments, got {}".format(len(args)))


class MatrixSSL(nn.Module):
    def __init__(self, backbone, emb_dim, gamma: float = 1.0, asym=True, momentum=0.9, loss_type: str = 'mce'):
        """
        backbone: encoder/embedding function
        emb_dim: embedding dimension 
        gamma: weight ratio in alignment loss (see alignment loss func)
        asym: whether we use asymmetric siamese (online-target) networks, with momentum averaging for the target network
        momentum: momentum parameter for moving average of target network. only required if asym = True. must be within [0, 1]
        loss_type: which loss to use for training. options are 'mce'/'mean_norm_diff'/'norm_mean_diff'
        """
        super().__init__()
        self.asym = asym

        self.gamma = gamma
        self.backbone = backbone
        self.emb_dim = emb_dim
        self.loss_type = loss_type
        if not asym and self.loss_type == 'mce':
            raise ValueError("MCE loss doesn't work with siamese networks")
        if asym:
            self.momentum = momentum

            self.online_backbone = copy.deepcopy(self.backbone)
            self.target_backbone = copy.deepcopy(self.backbone)

            # start with making everything linear, just one layer
            self.online_projector = nn.Sequential(
                nn.Linear(emb_dim, emb_dim, bias=False),
                nn.BatchNorm1d(emb_dim)
            )
            self.target_projector = nn.Sequential(
                nn.Linear(emb_dim, emb_dim, bias=False),
                nn.BatchNorm1d(emb_dim)
            )

            self.predictor = nn.Sequential(
                nn.Linear(emb_dim, emb_dim, bias=False),
                nn.BatchNorm1d(emb_dim)
            )
            #nn.Linear(emb_dim, emb_dim, bias=False) # only in online network
            # self.projector = nn.Sequential(
            #     nn.Linear(emb_dim, hidden_dim, bias=False), 
            #     nn.BatchNorm1d(hidden_dim),
            #     nn.Linear(hidden_dim, emb_dim, bias=False))
            
            # self.predictor = nn.Sequential(
            #     nn.Linear(emb_dim, emb_dim, bias=False))

            self.online = nn.Sequential(
                self.online_backbone, 
                self.online_projector,
                self.predictor
                )
            self.target = nn.Sequential(
                self.target_backbone, 
                self.target_projector
                )

            # ensure target network weights aren't updated via autograd
            for param in self.target.parameters():
                param.requires_grad = False

        else:
            self.online_backbone = copy.deepcopy(self.backbone)
            self.projector = nn.Sequential(
                nn.Linear(emb_dim, emb_dim, bias=False),
                nn.BatchNorm1d(emb_dim)
            )
            self.online = nn.Sequential(self.online_backbone, self.projector)

    def forward(self, x1, x2):
        if self.asym:
            z1, z2 = self.online(x1), self.online(x2)
            with torch.no_grad():
                p1, p2 = self.target(x1), self.target(x2)

            # correlation = True terms are uniformity loss terms
            # correlation = False are alignment terms
            if self.loss_type == 'mce':
                loss = (
                    mce_loss_func(p2, z1, correlation=True) +
                    mce_loss_func(p1, z2, correlation=True) +
                    self.gamma * mce_loss_func(p2, z1, correlation=False) +
                    self.gamma * mce_loss_func(p1, z2, correlation=False)
                    ) * 0.5
            elif self.loss_type == 'mean_norm_diff':
                loss = mean_norm_diff_loss(p2, z1) + uniformity_loss(p2, z1) +\
                       mean_norm_diff_loss(p1, z2) + uniformity_loss(p1, z2)
            elif self.loss_type == 'norm_mean_diff':
                loss = norm_mean_diff_loss(p2, z1) + uniformity_loss(p2, z1) +\
                       norm_mean_diff_loss(p1, z2) + uniformity_loss(p1, z2)
            return {'loss': loss, 'd_dict': []} #d_dict}
        else:
            f = self.online # backbone + projector
            z1, z2 = f(x1), f(x2)
            if self.loss_type == 'mean_norm_diff':
                loss = mean_norm_diff_loss(z1, z2) + uniformity_loss(z1, z2)
            elif self.loss_type == 'norm_mean_diff':
                loss = norm_mean_diff_loss(z1, z2) + uniformity_loss(z1, z2)
            return {'loss': loss, 'd_dict': []}


# %%
