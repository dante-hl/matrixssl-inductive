# %%
# %%
# set cwd set to matrixssl-inductive
import os
os.chdir(os.path.dirname(__file__))
from datetime import datetime
from data.loader import generate_cube_augs, generate_correlated_normal_augs
# %%
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import time

from torch.utils.data import TensorDataset, DataLoader

from models import Spectral, MatrixSSL

# %%

def conditional_arg_default(carg_name, args, set_inplace):
    """
    Checks if the conditional argument's default value exists. If it does, set the  default value in `args` or return the default value, depending on `set_inplace`. If the conditional argument's default value doesn't exist, function always returns None.

    carg_name: name of conditional argument
    set_inplace: (when the default value exists,) specifies whether to set the conditional argument to its default value in args, or whether to just return the default value
    """
    default_val = None
    nat_aug_defaults = {'d':25, 'k':5} # dict of cargs and default values for args in natural-augmentation data generating procedure
    corr_normal_aug_defaults = {'num_feats':5, 'feat_dim':5} # ditto for args in correlated normal augmentations generating procedure
    # general args
    if carg_name == 'momentum':
        if args.alg == 'mssla' or args.alg == 'mssls': # momentum required for mssl
            default_val = 0.9
    if carg_name == 'sgd_momentum':
        if args.optim == 'sgd': # sgd_momentum required for sgd
            default_val = 0.9
    # nat-aug args
    if carg_name in nat_aug_defaults:
        if args.aug == 'mult' or args.aug == 'add' or args.aug == 'corr':
            default_val = nat_aug_defaults[carg_name]
    if carg_name == 'tau_max':
        if args.aug == 'corr': # lower and upper bound for uniform sampling of tau needed for aug=corr
            default_val = 1
    # corr-normal args
    if carg_name in corr_normal_aug_defaults:
        if args.aug == 'normal':
            default_val = corr_normal_aug_defaults[carg_name]
    
    if default_val is not None: # if conditional arg has a default value
        if set_inplace: # set carg to default value in args
            setattr(args, carg_name, default_val)
            return
        else: # otherwise return the default value
            return default_val
    else:
        return None # for conditional args that don't have default values (i.e nat)
    

def parse_args():
    parser = argparse.ArgumentParser()

    # Algorithm, encoder, optimizer, embedding dimension
    parser.add_argument("--alg", type=str, required=True, 
    help="SSL algorithm to train, one of {'spectral', 'mssla', 'mssls'}. mssla, mssls refer to MatrixSSL with asymmetric (online, target) networks and a single network, respectively")
    parser.add_argument("--backbone", type=str, required=True, help="backbone architecture, one of {'linear', 'mlp'}")
    parser.add_argument("--optim", type=str, required=True, help="optimizer, one of {'sgd', 'adam'}")
    parser.add_argument("--emb_dim", type=int, default=10, help="embedding dimension")

    # Universal data generation arguments
    parser.add_argument("-n", type=int, default=(2 ** 16), help="number of data points")
    parser.add_argument("--aug", type=str, required=True, help="specifies augmentation scheme, one of {'mult', 'add', 'corr', 'normal'}. see generate_cube_data function in ./data/loader.py for more information")

    # Other training arguments
    parser.add_argument("--epochs", type=int, default=100, help="number training epochs")
    parser.add_argument("--bs", type=int, default=256, help="training minibatch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="optimizer learning rate hyperparam")
    parser.add_argument("--wd", type=float, default=1e-5, help="optimizer weight decay hyperparam")
    parser.add_argument("--sched", type=str, help="learning rate scheduler")

    # CONDITIONAL ARGUMENTS (everything below are conditional arguments in some way)
    parser.add_argument("--momentum", type=float, default=argparse.SUPPRESS, help="momentum averaging parameter for MatrixSSL with asymmetric networks. required if alg set to mssla, must be in [0, 1]")
    parser.add_argument("--sgd_momentum", type=float, default=argparse.SUPPRESS, help="SGD momentum parameter")
    # parser.add_argument("--hidden_dim", type=int, default=20, help="dimension of hidden layer in predictor network for MatrixSSL with assymetric networks. required if alg set to mssla")

    # Natural -> Augmentation Data Generation Parameters (for if aug = mult, add, corr)
    parser.add_argument("--nat", type=str, default=argparse.SUPPRESS, help="specifies natural data sampling scheme, one of {'bool', 'unif'}; see generate_cube_data function in ./data/loader.py for more information")
    parser.add_argument("-d", type=int, default=argparse.SUPPRESS, help="dimension of data")
    parser.add_argument("-k", type=int, default=argparse.SUPPRESS, help="denotes first k dimensions, which may be augmented or unaugmented, depending on the augment scheme")
    parser.add_argument("--tau_max", type=float, default=argparse.SUPPRESS, help="specifies bounds for uniformly sampling tau from [-tau_max, tau_max] when aug='corr'. must be >0")

    # Correlated Normal Data Generation Parameters (for if aug = normal)
    parser.add_argument("--num_feats", type=int, default=argparse.SUPPRESS, help="number of features in normal augmentation scheme")
    parser.add_argument("--feat_dim", type=int, default=argparse.SUPPRESS, help="dimension of each feature in normal augmentation scheme")

    # Saving/loading
    parser.add_argument("--save_dir", type=str, required=False, help="directory to save model weights/run details to")

    conditional_args = ['momentum', 'sgd_momentum', 'nat', 'd', 'k', 'tau_max', 'num_feats', 'feat_dim']
    exclude_from_runname_args = ['alg', 'backbone', 'optim', 'nat', 'save_dir']

    args = parser.parse_args()

    # Set conditional arguments to their defaults, if their input values are not already provided and if the conditions requiring them are satisfied
    for carg in conditional_args:
        if carg not in args:
            conditional_arg_default(carg, args, set_inplace=True)

    return parser, args, conditional_args, exclude_from_runname_args


def fit_classifier(linear, backbone, valloader, clf_loss_fn, clf_optim, device, clf_epochs=None):
    """
    Fits linear classifier on representations learned so far using validation dataset, outputs classification accuracy of classifier after fitting. Returns classification accuracy on validation set

    Ensure backbone has requires_gradients=False (and backbone set to eval()) before running

    linear: the linear classifier to fit
    backbone: encoder/embedding function
    valloader: DataLoader for validation set to fit on
    clf_loss_fn: linear classification loss function
    clf_optim: optimizer for linear classifier
    clf_epochs: number of training epochs (currently ignored, only using 1)
    """

    for idx, (x, y) in enumerate(valloader):
        x, y = x.to(device), y.to(device)
        # zero gradients
        clf_optim.zero_grad()
        # get linear classif predictor
        pred = torch.sigmoid(linear(backbone(x))).flatten()
        # calculate loss on labels, backprop
        clf_loss = clf_loss_fn(pred, y)
        clf_loss.backward()
        del x, y
        # update weights
        clf_optim.step()
    
    with torch.no_grad():
        total_count = 0
        correct_count = 0
        for idx, (x, y) in enumerate(valloader):
            x, y = x.to(device), y.to(device)
            pred = (torch.sigmoid(linear(backbone(x))) >= 0.5).flatten()
            correct_count += torch.sum(pred == y)
            total_count += len(y)
            del x, y
        clf_acc = correct_count / total_count
    return clf_acc

def generate_runpath(save_path:str, num=1):
    """
    Given directory path to save to, returns a compatible subdirectory name for the run that doesn't override existing run directories

    save_path: path to save directory to, without "_run#" appended
    returns: save_path with "_run#" appended
    """
    candidate_path = save_path + "_run" + str(num)
    if os.path.isdir(candidate_path):
        num += 1
        return generate_runpath(save_path, num)
    else:
        return candidate_path


def main():
    parser, args, cargs, xargs = parse_args()

    # forbid inconsistent arguments
    if hasattr(args, 'nat'):
        if (args.nat is not None) and args.aug == 'normal':
            raise Exception("'normal' augmentation scheme doesn't use natural data")
        if (args.nat is None) and (args.aug == 'mult' or args.aug == 'add' or args.aug == 'corr'):
            raise Exception("natural data should be specified if using 'mult', 'add', or 'args' augmentations")

    # Run data are organized in a hierarchy: at the top level, runs are organized by the setting of the SSL learning task (natural if not None, augmentation), then at the run level, where each run has its own directory, organized by hyperparameters in the run (alg, backbone, optim, emb_d etc.)

    # Thus, only SSL learning task arguments are included in the top level directories, while only run relevant arguments are included in the run level directories

    # CREATE LIST OF ARGS TO INCLUDE IN RUN (directory) NAME
    runname_args = []
    for arg_name, arg_value in vars(args).items():
        # always include alg, backbone, optim params
        if arg_name in ["alg", "backbone", "optim"]: 
            runname_args.append(str(arg_value))
            continue
        # include conditional arguments with non-default values in directory name. 
        if arg_name in cargs: 
            if (arg_name not in xargs) and (arg_value != conditional_arg_default(arg_name, args, set_inplace=False)):
                runname_args.append(f'{arg_name}={arg_value}')
            continue
        # include all other args with non-default values in directory name.
        if (arg_name not in xargs) and (arg_value != parser.get_default(arg_name)):
            runname_args.append(f'{arg_name}={arg_value}')

    # create list of 'essential arguments' - required task relevant hparams, required model hparams, any non-default arguments, and any optional arguments, if provided. for data visualization
    essential_args = []
    task_args = [getattr(args, attr) for attr in ['nat', 'aug'] if hasattr(args, attr)]
    essential_args.extend(task_args)
    essential_args.extend(runname_args)
    
    # CREATE SAVE DIRECTORY + PARENT (SSL task level) DIRECTORY (if needed) + RUN DIRECTORY

    # save_dir gives the option to create an even higher level folder to store directories containing runs. this may be helpful for when you want to more easily identify the outputs of specific scripts.
    if args.save_dir is not None: 
        if not os.path.isdir(args.save_dir):
            os.makedirs(args.save_dir)
        save_dir = args.save_dir
    else: # default is ./outputs
        save_dir = './outputs'
    # save to folder specified by natural, augmentation settings
    run_dir = os.path.join(save_dir, "_".join(task_args))
    # if not os.path.isdir(save_dir):
    #     os.makedirs(save_dir)
    # run name without "_run#" appended
    runname_prefix = "_".join(runname_args)
    # append "_run#" at end (start with 1 for first run)
    run_path = generate_runpath(os.path.join(run_dir, runname_prefix), 1)
    os.makedirs(run_path)

    # generate data
    if args.aug == 'normal': # normal correlated augmentations
        data_dict = generate_correlated_normal_augs(args.n, args.num_feats, args.feat_dim)
    else: # use natural-augmentation cube data procedure for data
        tau_max = None
        if args.aug == 'corr':
            tau_max = args.tau_max 
        data_dict = generate_cube_augs(args.n, args.d, args.k, args.nat, args.aug, tau_max)
    x1, x2 = data_dict["train"]

    # create dataloader
    trainset = TensorDataset(x1, x2)
    trainloader = DataLoader(trainset, batch_size=args.bs)

    print(f'Train Loader length: {len(trainloader)}')

    # define embedding dimension
    emb_d = args.emb_dim

    # initialize backbone, using embedding dimension determined above
    in_dim = args.d if hasattr(args, 'd') else args.feat_dim * args.num_feats
    if args.backbone == "linear":          
        backbone = nn.Linear(in_dim, emb_d)
    elif args.backbone == "mlp":
        backbone = nn.Sequential(
            nn.Linear(in_dim, 2*in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(2*in_dim, emb_d)
            )
    else:
        raise Exception("Invalid argument 'backbone', expected one of {'linear', 'mlp'}")

    # set up SSL architecture
    # ssl_model refers to both the backbone and the SSL algorithm/task used to train it
    if args.alg == "spectral":
        ssl_model = Spectral(backbone=backbone, emb_dim=emb_d)
    elif args.alg == "mssla":
        if (not (0 <= args.momentum <= 1)):
            raise Exception("Momentum averaging parameter must be set to a value within [0, 1] for asymmetric MatrixSSL")
        ssl_model = MatrixSSL(backbone=backbone, emb_dim=emb_d, asym=True, momentum=args.momentum)
    elif args.alg == "mssls":
        ssl_model = MatrixSSL(backbone=backbone, emb_dim=emb_d, asym=False)
    else:
        raise Exception("Invalid argument 'alg', must be one of {'spectral', 'mssla', 'mssls'}")
    # send model to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ssl_model = ssl_model.to(device)

    # define optimizer
    if args.optim == "sgd":
        opt = optim.SGD(ssl_model.parameters(), lr=args.lr, weight_decay=args.wd, momentum=args.sgd_momentum)
    elif args.optim == "adam":
        opt = optim.Adam(ssl_model.parameters(), lr=args.lr, weight_decay=args.wd)
    else:
        raise Exception("Invalid argument 'optim', must be one of {'sgd', 'adam'}")

    # define learning rate scheduler (if any)
    scheduler = None
    if hasattr(args, 'sched'):
        if args.sched == 'step':
            scheduler = StepLR(opt, step_size=30, gamma=0.1)
    # TODO: ############################################################################
    
    epochs = args.epochs

    # SET UP LOGGING
    train_losses = []
    gradient_norms = {}
    # for now, listen to only the gradient norms of the weight and bias of the embedding function (linear)
    gradient_norm_parts = ['weight', 'bias']
    # initialize empty lists for each part of gradient norm to log throughout train loop
    for part in gradient_norm_parts:
        gradient_norms[part] = []

    # TRAINING LOOP - See ssl_model class for specific forward passes
    for epoch in range(epochs):
        print(f"Epochs {epoch}")
        start_time = time.time()

        ssl_model.train()
        # listen to gradient calculations for online network
        for param in ssl_model.online.parameters():
            param.requires_grad = True

        # inner training loop
        for idx, (x1, x2) in enumerate(trainloader):
            x1, x2 = x1.to(device), x2.to(device)
            opt.zero_grad()
            # run forward() on ssl_model, which is expected to return a SSL loss on x1, x2
            # the forward function is specific to each SSL algorithm (Spectral, MatrixSSL, etc.)
            loss = ssl_model(x1, x2)['loss']
            # log training loss for each train step
            train_losses.append(loss.item())
            # backprop
            loss.backward()
            # calculate and log gradient norms
            for name, param in ssl_model.online_backbone.named_parameters():
                if param.grad is not None:
                    gradient_norm = torch.norm(param.grad)
                    gradient_norms[name].append(gradient_norm.item())
            del loss, x1, x2
            # update weights
            opt.step()
            # momentum averaging for mssl with asymmetric networks
            if getattr(ssl_model, "asym", False):
                with torch.no_grad():
                    # terminates at end of shortest iterator, excludes predictor weights
                    # need to ensure backbone and projector networks the same   
                    for online_param, target_param in zip(ssl_model.online.parameters(), ssl_model.target.parameters()):
                        target_param.mul_(args.momentum).add_((1-args.momentum) * online_param)

        # update learning rate as necessary
        if scheduler is not None:
            scheduler.step()
        # every 20 epochs, save model weights. include time of save.
        if (epoch + 1) % 20 == 0:
            current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            torch.save(ssl_model.online_backbone.state_dict(), os.path.join(run_path, f"model_weights_{current_time}"))

        # EVALUATE DOWNSTREAM LIN CLF PERFORMANCE AT END OF EACH EPOCH 
        # ######### (NO LONGER WORKS, SINCE REMOVED VALIDATION SPLIT) ##############
        # ssl_model.eval()
        # # stop calculating gradients for backbone in online network
        # for param in ssl_model.online_backbone.parameters():
        #     param.requires_grad = False

        # # instantiate linear classifier
        # linear = nn.Linear(ssl_model.emb_dim, 1).to(device)
        # # fit linear classifier, print classification accuracy on validation set
        # clf_acc = fit_classifier(
        #     linear,
        #     ssl_model.online_backbone,
        #     valloader,
        #     nn.BCELoss(),
        #     optim.SGD(linear.parameters(), lr=0.1, momentum=0.9),
        #     device=device
        #     )
        # del linear
        # print(f'Epoch {epoch+1} classification accuracy: {clf_acc}')
        # val_accs.append(clf_acc)

        end_time = time.time()
        print(f'Epoch time: {end_time - start_time}')

    # store final model weights, data, args, metrics run_path 
    run_dict = {
        "model_weights":ssl_model.online_backbone.state_dict(),
        "data": data_dict,
        "optim": args.optim,
        "args": args,
        "essential_args": essential_args,
        "train_losses":train_losses,
        "gradient_norms":gradient_norms
    }
    torch.save(run_dict, os.path.join(run_path, 'run_dict'))


if __name__ == '__main__':
    main()

# %%
