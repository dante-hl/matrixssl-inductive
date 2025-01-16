"""
    The code is mainly based on https://github.com/xinliu20/MEC
"""

import argparse
import builtins
import json
import math
import os
import random
import shutil
import time
import warnings

from tqdm import tqdm
import wandb

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

import torch.nn.functional as F
import numpy as np

from models.matrixssl import centering_matrix
from models.mec import MEC, SpectralMEC
from models.cifar_resnet import cifar_resnet34, cifar_resnet50
import data.augments

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

IS_MASTER = False
SHOULD_LOG = False

parser = argparse.ArgumentParser(description='PyTorch CIFAR Training')

parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--dataset', default='CIFAR10', help='dataset')
parser.add_argument('--save_dir', default='', type=str, help='save dir')
parser.add_argument('--project_name', default='project_name', type=str, help='wandb project name')
parser.add_argument('--run_name', default='run_name', type=str, help='wandb run name')
parser.add_argument('--wandb_logging', choices=['none', 'master', 'all'], default='none', help="wandb logging, if using wandb at all")

parser.add_argument('--model', help="model type, one of {'mec', 'spectralmec'}")
parser.add_argument('--loss_type', help="loss to use, only if model == mec. one of {'mce', 'mean_norm_diff', 'norm_mean_diff'}")
parser.add_argument('--layer_type', help="added layer to use, only if model == spectralmec. one of {'fixed', 'learnt'}")
parser.add_argument('--asym', action='store_true', help="whether to use asymmetric networks for spectralmec. if False, use same network for both augs")


parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=1024, type=int,
                    metavar='N',
                    help='mini-batch size (default: 1024), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.5, type=float,
                    metavar='LR', help='initial (base) learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=1e-5, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=100, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

parser.add_argument('--dim', default=2048, type=int,
                    help='feature dimension (default: 2048)')
parser.add_argument('--correlation', default=True, type=bool,
                    help='use correlation mce or not')
parser.add_argument('--logE', default=False, type=bool,
                    help='use logE or not')
parser.add_argument('--pred_dim', default=512, type=int,
                    help='hidden dimension of the predictor (default: 512)')
parser.add_argument('--m', default=1024, type=float, metavar='M',
                    help='the total batch size across all GPUs')
parser.add_argument('--mu', default=None, type=float, metavar='M',
                    help='(m + d) / 2 of Eqn.2')
parser.add_argument('--eps', default=64, type=float, metavar='M',
                    help='square of the upper bound of the expected decoding error')
parser.add_argument('--mce_lambd', default=0.5, type=float, metavar='M',
                    help='mce lambda')
parser.add_argument('--mce_mu', default=0.5, type=float, metavar='M',
                    help='mce mu')
parser.add_argument('--mce_order', default=4, type=int, metavar='M',
                    help='mce order')
parser.add_argument('--gamma', default=1.0, type=float, metavar='M',
                    help='mce gamma')
parser.add_argument('--align_gamma', default=0.003, type=float, help='align_gamma')
parser.add_argument('--teacher_momentum', default=0.996, type=float, metavar='M',
                    help='momentum of teacher update')

def main():

    args = parser.parse_args()
        
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu
    
    # suppress printing if process not master
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
        torch.distributed.barrier()

    # set global master process flag
    global IS_MASTER
    IS_MASTER = (not args.multiprocessing_distributed) or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0)

    # set flag for if process should log (False when wandb_logging = 'none' by default)
    global SHOULD_LOG 
    SHOULD_LOG = (args.wandb_logging == 'all') or (args.wandb_logging == 'master' and IS_MASTER)

    if SHOULD_LOG:
        wandb.init(project=args.project_name,
                   entity="dantehl", config=args.__dict__,
                   name=f"{args.run_name}_gpu{gpu}" if args.wandb_logging == 'all' else args.run_name
        )

    if args.dataset == 'ImageNet':
        base = models.__dict__[args.arch]
    elif args.dataset == 'CIFAR10':
        if args.arch == 'resnet34':
            base = cifar_resnet34
        elif args.arch == 'resnet50':
            base = cifar_resnet50
        else:
            raise Exception(f'architecture {args.arch} not supported for cifar10')

    # create model
    print("=> creating model '{}'".format(args.arch))
    if args.model == 'mec':
        model = MEC(
            base, args.dim, args.pred_dim, loss_type=args.loss_type
            )
    elif args.model == 'spectralmec':
        model = SpectralMEC(
            base, args.dim, args.pred_dim, layer_type=args.layer_type, asym=args.asym, momentum=args.teacher_momentum
        )
    else:
        raise Exception(f'invalid model argument, received {args.model}')

    # infer learning rate before changing batch size
    total_batch_size = args.batch_size
    init_lr = args.lr * args.batch_size / 256

    if args.distributed:
        # Apply SyncBN
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        # comment out the following line for debugging
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    else:
        # this code only supports DistributedDataParallel.
        raise NotImplementedError("Only DistributedDataParallel is supported.")

    # define SGD optimizer
    optim_params = model.parameters()
    optimizer = torch.optim.SGD(optim_params, init_lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    if args.dataset == 'ImageNet':
        traindir = os.path.join(args.data, 'train')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
        
        # follow MoCov3's augmentation recipe
        augmentation1 = [
            transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([data.augments.GaussianBlur([.1, 2.])], p=1.0),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]

        augmentation2 = [
            transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([data.augments.GaussianBlur([.1, 2.])], p=0.1),
            transforms.RandomApply([data.augments.Solarize()], p=0.2),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]

        train_dataset = datasets.ImageFolder(
            traindir,
            data.augments.TwoCropsTransform(transforms.Compose(augmentation1), transforms.Compose(augmentation2))
            )
        
    elif args.dataset == 'CIFAR10':
        normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                              std=[0.2470, 0.2435, 0.2616])
        
        cifar_augments = [
                transforms.RandomResizedCrop(size=32, scale=(0.2, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply([
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
                ], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([
                    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
                ], p=0.5),
                transforms.RandomSolarize(threshold=128, p=0.1),
                transforms.ToTensor(),
            ]
        # cifar_augments = [
        #     transforms.RandomCrop(32, padding=4),
        #     transforms.RandomHorizontalFlip(),
        #     transforms.ToTensor(),
        #     normalize
        # ]

        train_dataset = datasets.CIFAR10(
            args.data, train=True, 
            transform=data.augments.TwoCropsTransform(transforms.Compose(cifar_augments), transforms.Compose(cifar_augments)), download=True
            )
    else:
        raise NotImplementedError

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)

    momentum_schedule = cosine_scheduler(args.teacher_momentum, 1,
                                         args.epochs, len(train_loader))

    lr_schedule = cosine_scheduler(init_lr, 1e-6, # changed from 0
                                   args.epochs, len(train_loader), warmup_epochs=10)

    # following the notations of the paper
    args.m = total_batch_size
    d = args.dim
    args.mu = (args.m + d) / 2 # not really used
    eps_d = args.eps / d
    lamda = 1 / (args.m * eps_d)
    # warm up of lamda (lamda_inv) to ensure the convergence of Taylor expansion (Eqn.2)
    lamda_schedule = lamda_scheduler(8/lamda, 1/lamda, args.epochs, len(train_loader), warmup_epochs=10)

    # # Assuming your model is called 'model'
    # for name, param in model.module.projector.named_parameters():
    #     print(f"Parameter {name}: device = {param.device}")
        
    # # You can also check the device of each submodule
    # for i, module in enumerate(model.module.projector):
    #     print(f"\nModule {i}: {type(module).__name__}")
    #     for name, param in module.named_parameters():
    #         print(f"  Parameter {name}: device = {param.device}")

    # # Check the new_layer parameters
    # print("\nNew Layer:")
    # for name, param in model.module.new_layer.named_parameters():
    #     print(f"Parameter {name}: device = {param.device}")

    # # If using asymmetric setup, check teacher network
    # if model.module.asym:
    #     print("\nTeacher Network:")
    #     for name, param in model.module.teacher.named_parameters():
    #         print(f"Parameter {name}: device = {param.device}")
        
    #     print("\nPredictor:")
    #     for name, param in model.module.predictor.named_parameters():
    #         print(f"Parameter {name}: device = {param.device}")


    for epoch in range(args.start_epoch, args.epochs):
        print(f'Epoch {epoch}')
        if args.distributed:
            train_sampler.set_epoch(epoch)

        # train for one epoch
        train(train_loader, model, optimizer, epoch, lr_schedule, momentum_schedule, lamda_schedule, args)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            if epoch % 50 == 49:
                save_checkpoint(
                    {
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'optimizer' : optimizer.state_dict(),
                    'dim' : args.dim
                    }, 
                    is_best=False,
                    save_dir=args.save_dir,
                    filename='checkpoint_{:04d}.pth.tar'.format(epoch))


def train(train_loader, model, optimizer, epoch, lr_schedule, momentum_schedule, lamda_schedule, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4f')
    align_losses = AverageMeter('Align Loss', ':.4f')
    unif_losses = AverageMeter('Unif Loss', ':.4f')
    # mec = AverageMeter('MEC', ':6.3f')
    # mce = AverageMeter('MCE', ':6.3f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode

    eigen_z1 = []
    eigen_z2 = []

    #### TEMP
    def print_memory_stats():
        print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB\n")

    model.train()
    end = time.time()
    for i, (images, _) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        it = len(train_loader) * epoch + i
        cur_lr = lr_schedule[it]
        lamda_inv = lamda_schedule[it]
        momentum = momentum_schedule[it] # this is moving average momentum, not optimizer momentum

        # update the learning rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = cur_lr

        # print('Before .cuda')
        # print_memory_stats()  #####

        if args.gpu is not None:
            images[0] = images[0].cuda(args.gpu, non_blocking=True)
            images[1] = images[1].cuda(args.gpu, non_blocking=True)

        # print('After .cuda')
        # print_memory_stats()  #####

        # compute output and loss
        torch.cuda.empty_cache()
        loss_dict = model(x1=images[0], x2=images[1])
        loss = loss_dict['loss']

        # print('After forward')
        # print_memory_stats()  #####
        losses.update(loss.detach().item(), images[0].size(0))
        if args.model == 'mec':
            alignment_loss = loss_dict['alignment_loss']
            unif_loss = loss_dict['uniformity_loss']
            align_losses.update(alignment_loss.detach().item(), images[0].size(0))
            unif_losses.update(unif_loss.detach().item(), images[0].size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print('After loss.backward., opt.step')
        # print_memory_stats()  #####

        # momentum averaging for mssl with asymmetric networks
        if getattr(model.module, "asym", False):
            # if i < 10:
                 # print('Momentum average is being done!')
            with torch.no_grad():
                # terminates at end of shortest iterator, excludes predictor weights
                # need to ensure backbone and projector networks the same  
                # should or should not be model.module.online.parameters() ??? used module in older code 
                for online_param, teacher_param in zip(model.module.online.parameters(), model.module.teacher.parameters()):
                    teacher_param.mul_(momentum).add_((1-momentum) * online_param)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

    if SHOULD_LOG:
        if args.model == 'mec':
            wandb.log({
                    "loss": losses.avg,
                    "alignment_loss": align_losses.avg,
                    "uniformity_loss": unif_losses.avg,
                })
        else:
            wandb.log({
                    "loss": losses.avg,
            })



# Taylor expansion
def matrix_log(Q, order=4):
    n = Q.shape[0]
    Q = Q - torch.eye(n).detach().to(Q.device)
    cur = Q
    res = torch.zeros_like(Q).detach().to(Q.device)
    for k in range(1, order + 1):
        if k % 2 == 1:
            res = res + cur * (1. / float(k))
        else:
            res = res - cur * (1. / float(k))
        cur = cur @ Q

    return res


def mce_loss_func(p, z, lamda=1., mu=1., order=4, align_gamma=0.003, correlation=True, logE=False):
    p = gather_from_all(p)
    z = gather_from_all(z)

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

# only referenced in mec loss, comment out if not needed
def loss_func(p, z, lamda_inv, order=4):

    p = gather_from_all(p)
    z = gather_from_all(z)

    p = F.normalize(p)
    z = F.normalize(z)

    c = p @ z.T

    c = c / lamda_inv 

    power_matrix = c
    sum_matrix = torch.zeros_like(power_matrix)

    for k in range(1, order+1):
        if k > 1:
            power_matrix = torch.matmul(power_matrix, c)
        if (k + 1) % 2 == 0:
            sum_matrix += power_matrix / k
        else: 
            sum_matrix -= power_matrix / k

    trace = torch.trace(sum_matrix)

    return trace


def save_checkpoint(state, is_best, save_dir='', filename='checkpoint.pth.tar'):
    """
    creates 'pretrain' subdirectory below save_dir, to distinguish from outputs of linear.sh
    """
    exact_dir = os.path.join(save_dir, 'pretrain')
    os.makedirs(exact_dir, exist_ok=True)
    checkpoint_path = os.path.join(exact_dir, filename)
    torch.save(state, checkpoint_path)
    if is_best:
        model_best_path = os.path.join(exact_dir, 'model_best.pth.tar')
        shutil.copyfile(filename, model_best_path)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

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


def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule


def lamda_scheduler(start_warmup_value, base_value, epochs, niter_per_ep, warmup_epochs=5):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    schedule = np.ones(epochs * niter_per_ep - warmup_iters) * base_value
    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule


if __name__ == '__main__':
    main()
