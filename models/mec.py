import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import copy
from .matrixssl import centering_matrix, matrix_log, mean_norm_diff_loss, norm_mean_diff_loss, uniformity_loss
# TODO may have to make distributed for mnd and nmd losses, so may have to just uncomment below.. read online   
from .spectral import D # <-- spectral contrastive loss

#### TEMP
def print_memory_stats():
    print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    print(f"Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB\n")

# uncomment if importing doesn't work
# def centering_matrix(b: int):
#     """
#     Return centering matrix of dimension b
#     """
#     return torch.eye(b) - (torch.ones((b, b)) / b)

# def matrix_log(X: torch.tensor, order: int = 4):
#     """
#     matrix_log, from MEC. uses taylor expansion approximation
#     """
#     device = X.device
#     assert X.shape[0] == X.shape[1]
#     d = X.shape[0]
#     series = torch.zeros_like(X).detach().to(device)
#     I_d = torch.eye(d).to(device)
#     mult = X - I_d
#     summand = mult
#     for i in range(1, order + 1):
#         if i % 2 == 1:
#             series = series + (1. / float(i)) * summand
#         else:
#             series = series - (1. / float(i)) * summand
#         summand = summand @ mult
#     del mult, I_d
#     return series

class GatherLayer(torch.autograd.Function):
    """
    Gather tensors from all workers with support for backward propagation:
    This implementation does not cut the gradients as torch.distributed.all_gather does.
    """

    @staticmethod
    def forward(ctx, x):
        output = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        dist.all_reduce(all_gradients)
        return all_gradients[dist.get_rank()]

def gather_from_all(tensor: torch.Tensor) -> torch.Tensor:
    """
    Similar to classy_vision.generic.distributed_util.gather_from_all
    except that it does not cut the gradients
    """
    if tensor.ndim == 0:
        # 0 dim tensors cannot be gathered. so unsqueeze
        tensor = tensor.unsqueeze(0)

    gathered_tensors = GatherLayer.apply(tensor)

    gathered_tensor = torch.cat(gathered_tensors, 0)

    return gathered_tensor

def mce_loss_func(p, z, lamda=1., mu=1., order=4, align_gamma=0.003, correlation=True, logE=False):
    #########################################################
    # REMOVE COMMENT WHEN RUNNING DISTRIBUTED TRAINING
    #########################################################
    # p = gather_from_all(p)
    # z = gather_from_all(z)

    p = F.normalize(p)
    z = F.normalize(z)

    m = z.shape[0]
    n = z.shape[1]
    # print(m, n)
    J_m = centering_matrix(m).detach().to(z.device)
    
    if correlation:
        P = lamda * torch.eye(n).to(z.device)
        Q = (1. / m) * (p.T @ J_m @ z) + mu * torch.eye(n).to(z.device)
    else:
        P = (1. / m) * (p.T @ J_m @ p) + mu * torch.eye(n).to(z.device)
        Q = (1. / m) * (z.T @ J_m @ z) + mu * torch.eye(n).to(z.device)
    
    return torch.trace(- P @ matrix_log(Q, order))


class MEC(nn.Module):
    """
    Build a MEC model.
    """
    def __init__(self, base_encoder, dim=2048, pred_dim=512, momentum=0.9, loss_type='mce', 
                 mce_gamma=1.0, mce_lambda=0.5, mce_mu=0.5, mce_order=4):
        """
        dim: feature dimension (default: 2048)
        pred_dim: hidden dimension of the predictor (default: 512)
        """
        super(MEC, self).__init__()
        self.momentum = momentum # not actually used when (moving average) momentum scheduling used
        self.loss_type = loss_type
        self.mce_gamma = mce_gamma # coefficient multiplying (alignment/uniformity terms)
        self.mce_lambda = mce_lambda # coefficient multiplying matrix P in correlation=True terms
        self.mce_mu = mce_mu # coefficient for extra identity term added for stability in each matrix in loss
        self.mce_order = mce_order # order of matrix log
        self.asym = True # MEC is automatically asymmeytric in nature

        # create base encoder
        # num_classes is the output fc dimension, zero-initialize last BNs
        self.encoder = base_encoder(num_classes=dim, zero_init_residual=True)
        
        # create 3 layer projector (involves adding 'fc' from encoder)
        prev_dim = self.encoder.fc.weight.shape[1]
        self.projector = nn.Sequential(nn.Linear(prev_dim, prev_dim, bias=False),
                                        nn.BatchNorm1d(prev_dim),
                                        nn.ReLU(inplace=True), # first layer
                                        nn.Linear(prev_dim, prev_dim, bias=False),
                                        nn.BatchNorm1d(prev_dim),
                                        nn.ReLU(inplace=True), # second layer
                                        self.encoder.fc,
                                        nn.BatchNorm1d(dim, affine=False)) # output layer
        self.projector[6].bias.requires_grad = False # hack: not use bias as it is followed by BN

        # effectively remove fc from 'encoder' for clean separation
        self.encoder.fc = nn.Identity()

        # create 2 layer predictor
        self.predictor = nn.Sequential(nn.Linear(dim, pred_dim, bias=False),
                                        nn.BatchNorm1d(pred_dim),
                                        nn.ReLU(inplace=True), # hidden layer
                                        nn.Linear(pred_dim, dim)) # output layer

        self.teacher_encoder = copy.deepcopy(self.encoder)
        self.teacher_projector = copy.deepcopy(self.projector)
        
        # used as references for momentum averaging
        self.online = nn.ModuleList([self.encoder, self.projector]) # encoder, projector 
        self.teacher = nn.ModuleList([self.teacher_encoder, self.teacher_projector]) # encoder, projector
        
        for p in self.teacher.parameters():
            p.requires_grad = False


    def forward(self, x1, x2):
        """
        Input:
            x1: first views of images
            x2: second views of images
        Output:
            p1, p2, z1, z2: targets and predictions of teacher and student networks, respectively
        """
        z1 = F.normalize(self.encoder(x1), p=2, dim=1)
        z2 = F.normalize(self.encoder(x2), p=2, dim=1)
        z1 = self.predictor(self.projector(z1))
        z2 = self.predictor(self.projector(z2))

        # print(f"After z1, z2: {z1.shape}")
        # print_memory_stats()

        with torch.no_grad():
            p1 = self.teacher_projector(F.normalize(self.teacher_encoder(x1), p=2, dim=1))
            p2 = self.teacher_projector(F.normalize(self.teacher_encoder(x2), p=2, dim=1))

        # print("After p1, p2")
        # print_memory_stats()

        if self.loss_type == 'mce':
            alignment_loss = (
                self.mce_gamma * mce_loss_func(p2, z1, correlation=False, lamda=self.mce_lambda, mu=self.mce_mu, order=self.mce_order) +
                self.mce_gamma * mce_loss_func(p1, z2, correlation=False, lamda=self.mce_lambda, mu=self.mce_mu, order=self.mce_order)
            )
            unif_loss = (
                mce_loss_func(p2, z1, correlation=True, lamda=self.mce_lambda, mu=self.mce_mu, order=self.mce_order) +
                mce_loss_func(p1, z2, correlation=True, lamda=self.mce_lambda, mu=self.mce_mu, order=self.mce_order)
            )
            loss = 0.5 * (alignment_loss + unif_loss)
            # loss = (
            #         mce_loss_func(p2, z1, correlation=True, lamda=self.mce_lambda, mu=self.mce_mu, order=self.mce_order) +
            #         mce_loss_func(p1, z2, correlation=True, lamda=self.mce_lambda, mu=self.mce_mu, order=self.mce_order) +
            #         self.mce_gamma * mce_loss_func(p2, z1, correlation=False, lamda=self.mce_lambda, mu=self.mce_mu, order=self.mce_order) +
            #         self.mce_gamma * mce_loss_func(p1, z2, correlation=False, lamda=self.mce_lambda, mu=self.mce_mu, order=self.mce_order)
            #         ) * 0.5
        elif self.loss_type == 'mean_norm_diff':
            alignment_loss = mean_norm_diff_loss(p2, z1) + mean_norm_diff_loss(p1, z2)
            unif_loss = uniformity_loss(p2, z1) + uniformity_loss(p1, z2)
            loss = alignment_loss + unif_loss
            # loss = mean_norm_diff_loss(p2, z1) + uniformity_loss(p2, z1) +\
            #        mean_norm_diff_loss(p1, z2) + uniformity_loss(p1, z2)
        elif self.loss_type == 'norm_mean_diff':
            alignment_loss = norm_mean_diff_loss(p2, z1) + norm_mean_diff_loss(p1, z2)
            unif_loss = uniformity_loss(p2, z1) + uniformity_loss(p1, z2)
            loss = alignment_loss + unif_loss
            # loss = norm_mean_diff_loss(p2, z1) + uniformity_loss(p2, z1) +\
            #        norm_mean_diff_loss(p1, z2) + uniformity_loss(p1, z2)
        else:
            raise Exception(f"loss_type not specified, was {self.loss_type}")
            
        # print("After loss")
        # print_memory_stats()
        return {'loss': loss, 'alignment_loss':alignment_loss, 'uniformity_loss':unif_loss, 'd_dict': []} #, {'z1':z1, 'z2':z2, 'p1':p1.detach(), 'p2':p2.detach()}
        

class FixedQuadraticLayer(nn.Module):
    """
    Custom neural net layer for fixed quadratic feature mapping. Given vector [x1, .... xd], returns [ x1^2, ... xd^2, x2x1, x3x1, x3x2, .... xd x(d-1) ].
    """
    def __init__(self, in_dim):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = int(self.in_dim * (self.in_dim + 1) / 2)

    def forward(self, x):
        """
        x: (..., d) torch tensor
        returns out: (..., d*(d+1)/2 ) tensor
        """
        device = x.device
        batch_dims = x.shape[:-1]
        out = torch.zeros(*batch_dims, self.out_dim).to(device)
        batched_outer_prods = torch.bmm(x.unsqueeze(-1), x.unsqueeze(-2))
        out[..., :self.in_dim] = torch.diagonal(batched_outer_prods, dim1=-2, dim2=-1)
        indices = torch.tril_indices(self.in_dim, self.in_dim, offset=-1).to(device)
        out[..., self.in_dim:] = batched_outer_prods[..., indices[0], indices[1]]
        return out
        
# TODO: update with all changes
class SpectralMEC(nn.Module):
    """
    MEC with Spectral Loss modification (extra layer, and use SCL loss)
    """
    def __init__(self, base_encoder, dim=2048, pred_dim=512, layer_type='learnt', asym=True, momentum=None):
        """
        dim: feature dimension (default: 2048)
        pred_dim: hidden dimension of the predictor (default: 512)
        layer_type: determines the 'extra layer' added. 'learnt' means randomly initialized and trained, 'fixed' means fixed to quadratic mapping of previous layer features
        asym: whether or not to use asymmetric setup (like in MatrixSSL). if True, then Spectral Contrastive Loss is symmetrized.
        """
        super(SpectralMEC, self).__init__()
        
        self.layer_type = layer_type
        if asym:
            assert momentum is not None, "momentum averaging required for asymmetric networks"
            self.momentum = momentum
        self.asym = asym
        
        # create base encoder
        # num_classes is the output fc dimension, zero-initialize last BNs
        self.encoder = base_encoder(num_classes=dim, zero_init_residual=True)
        prev_dim = self.encoder.fc.weight.shape[1]
        new_dim = int(prev_dim * (prev_dim + 1) / 2)

        # create new feature layer
        if self.layer_type == 'learnt':
            self.new_layer = nn.Linear(prev_dim, new_dim)
        elif self.layer_type == 'fixed': # use fixed qudaratic feature mapping
            self.new_layer = FixedQuadraticLayer(prev_dim)
        else:
            raise Exception(f"invalid input for 'layer_type', expected 'learnt' or 'fixed', got {layer_type}")

        # create 3 layer projector
        # self.projector = nn.Sequential(nn.Linear(prev_dim, prev_dim, bias=False),
        #                                 nn.BatchNorm1d(prev_dim),
        #                                 nn.ReLU(inplace=True), # first layer
        #                                 nn.Linear(prev_dim, prev_dim, bias=False),
        #                                 nn.BatchNorm1d(prev_dim),
        #                                 nn.ReLU(inplace=True), # second layer
        #                                 self.encoder.fc,
        #                                 nn.BatchNorm1d(dim, affine=False)) # output layer
        # self.projector[6].bias.requires_grad = False # hack: not use bias as it is followed by BN

        # # remove fc from 'encoder' for cleaner separation
        # self.encoder = nn.Sequential(*(list(self.encoder.children())[:-1])) 

        # # if symmetric, use only online (encoder > new layer > projector) on both augments
        # self.online = nn.Sequential(self.encoder, self.new_layer, self.projector)

        # create 3 layer projector (involves adding 'fc' from encoder)
        prev_dim = self.encoder.fc.weight.shape[1]
        self.projector = nn.Sequential(nn.Linear(new_dim, prev_dim, bias=False),
                                        nn.BatchNorm1d(prev_dim),
                                        nn.ReLU(inplace=True), # first layer
                                        nn.Linear(prev_dim, prev_dim, bias=False),
                                        nn.BatchNorm1d(prev_dim),
                                        nn.ReLU(inplace=True), # second layer
                                        self.encoder.fc,
                                        nn.BatchNorm1d(dim, affine=False)) # output layer
        self.projector[6].bias.requires_grad = False # hack: not use bias as it is followed by BN

        # effectively remove fc from 'encoder' for clean separation
        self.encoder.fc = nn.Identity()

        if self.asym:

            # create 2 layer predictor
            self.predictor = nn.Sequential(nn.Linear(dim, pred_dim, bias=False),
                                            nn.BatchNorm1d(pred_dim),
                                            nn.ReLU(inplace=True), # hidden layer
                                            nn.Linear(pred_dim, dim)) # output layer

            self.teacher_encoder = copy.deepcopy(self.encoder)
            self.teacher_new_layer = copy.deepcopy(self.new_layer)
            self.teacher_projector = copy.deepcopy(self.projector)
        
            # used as references for momentum averaging
            self.online = nn.ModuleList([self.encoder, self.new_layer, self.projector]) # encoder, projector 
            self.teacher = nn.ModuleList([self.teacher_encoder, self.teacher_new_layer, self.teacher_projector]) # encoder, projector
        
            for p in self.teacher.parameters():
                p.requires_grad = False

        # if self.asym: # include teacher, predictor in student
        #     # 2-layer predictor
        #     self.predictor = nn.Sequential(nn.Linear(dim, pred_dim, bias=False),
        #                                     nn.BatchNorm1d(pred_dim),
        #                                     nn.ReLU(inplace=True), # hidden layer
        #                                     nn.Linear(pred_dim, dim)) # output layer
    
        #     self.teacher = copy.deepcopy(self.online) # encoder > new layer > projector
        #     for p in self.teacher.parameters():
        #         p.requires_grad = False

        # during evaluations, create a new copy of resnet, init final fc and freeze all others. my guess is that you then just load state for all frozen resnet50/base encoder layers

    def to(self, device):
        """Ensures all submodules are moved to the specified device"""
        super().to(device)
        if self.asym:
            self.online.to(device)
            self.teacher.to(device)
            self.predictor.to(device)
        return self


    def forward(self, x1, x2):
        """
        Input:
            x1: first views of images
            x2: second views of images
        Output:
            loss and d_dict (parts of loss, in a dictionary), and:
            if asymmetric: 
                p1, p2, z1, z2: targets and predictions of teacher and student/online networks respectvely
            if symmetric:
                z1, z2: outputs of online network on augmentations x1, x2
        """
        print(f"x1 device: {x1.device}")
        print(f"x2 device: {x2.device}")
        # Continue with rest of forward pass...

        if self.asym:
            z1 = F.normalize(self.encoder(x1), p=2, dim=1)
            z2 = F.normalize(self.encoder(x2), p=2, dim=1)
            z1 = self.predictor(self.projector(self.new_layer(z1)))
            z2 = self.predictor(self.projector(self.new_layer(z2)))

            with torch.no_grad():
                p1 = self.teacher_projector(self.new_layer(F.normalize(self.teacher_encoder(x1), p=2, dim=1)))
                p2 = self.teacher_projector(self.new_layer(F.normalize(self.teacher_encoder(x2), p=2, dim=1)))

            # symmetrize SCL loss:
            loss1, d_dict1 = D(z1, p2)
            loss2, d_dict2 = D(z2, p1)
            loss = 0.5 * (loss1 + loss2)
            return {'loss': loss, 'd_dict': [d_dict1, d_dict2]} #, {'z1':z1, 'z2':z2, 'p1':p1.detach(), 'p2':p2.detach()}
            
        else: # only use online network, # TODO: fix later
            z1, z2 = self.online(x1), self.online(x2)
            loss, d_dict = D(z1, z2)
            return {'loss': loss, 'd_dict': [d_dict]} #, {'z1':z1, 'z2':z2}
        
