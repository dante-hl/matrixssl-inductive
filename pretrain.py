# %%
# %%
# set cwd set to matrixssl-inductive
import os
os.chdir(os.path.dirname(__file__))
from datetime import datetime
from data.loader import generate_cube_data
# %%
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import time

from torch.utils.data import TensorDataset, DataLoader

from models import Spectral, MatrixSSL

# %%

def conditional_arg_default(carg_name, args, set_inplace):
    """
    Set default value for input conditional argument name, if needed

    carg_name: name of conditional argument
    """
    default_val = None
    if carg_name == 'momentum':
        if args.alg == 'mssla' or args.alg == 'mssls': # momentum required for mssl
            default_val = 0.9
    if carg_name == 'tau_max':
        if args.aug == 'corr': # lower and upper bound for uniform sampling of tau needed for aug=corr
            default_val = 1
    if default_val is not None:
        if set_inplace:
            setattr(args, carg_name, default_val)
            return
    return default_val # if getting default, return None if args don't satisfy conditions, or return default value if they do
    

def parse_args():
    parser = argparse.ArgumentParser()

    # Algorithm, encoder, optimizer, embedding dimension
    parser.add_argument("--alg", type=str, required=True, 
    help="SSL algorithm to train, one of {'spectral', 'mssla', 'mssls'}. mssla, mssls refer to MatrixSSL with asymmetric (online, target) networks and a single network, respectively")
    parser.add_argument("--backbone", type=str, required=True, help="backbone architecture, one of {'linear', 'mlp'}")
    parser.add_argument("--optim", type=str, required=True, help="optimizer, one of {'sgd', 'adam'}")
    parser.add_argument("--emb_dim", type=int, default=10, help="embedding dimension")

    # Data generation arguments
    parser.add_argument("--nat", type=str, required=True, help="specifies natural data sampling scheme, one of {'bool', 'unif'}; see generate_cube_data function in ./data/loader.py for more information")
    parser.add_argument("--aug", type=str, required=True, help="specifies augmentation scheme, one of {'mult', 'add'}. see generate_cube_data function in ./data/loader.py for more information")
    parser.add_argument("--label", type=str, required=True, help="specifies labeling scheme. see generate_cube_data function in ./data/loader.py for more information")

    parser.add_argument("-n", type=int, default=(2 ** 16) + 12500, help="number of data points")
    parser.add_argument("-v", type=int, default=12500, help="size of validation set, input some v<n")
    parser.add_argument("-d", type=int, default=25, help="dimension of data")
    parser.add_argument("-k", type=int, default=5, help="number invariant dimensions")

    # Other training arguments
    parser.add_argument("--epochs", type=int, default=100, help="number training epochs")
    parser.add_argument("--bs", type=int, default=256, help="training minibatch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="optimizer learning rate hyperparam")
    parser.add_argument("--wd", type=float, default=1e-5, help="optimizer weight decay hyperparam")

    # Conditional arguments 
    parser.add_argument("--momentum", type=float, default=argparse.SUPPRESS, help="momentum averaging parameter for MatrixSSL with asymmetric networks. required if alg set to mssla, must be in [0, 1]")
    # parser.add_argument("--hidden_dim", type=int, default=20, help="dimension of hidden layer in predictor network for MatrixSSL with assymetric networks. required if alg set to mssla")
    parser.add_argument("--tau_max", type=float, default=argparse.SUPPRESS, help="specifies bounds for uniformly sampling tau from [-tau_max, tau_max] when aug='corr'. must be >0")
    conditional_args = ['momentum', 'tau_max']

    # Saving/loading
    parser.add_argument("--save_dir", type=str, required=False, help="directory to save model weights/run details to")

    args = parser.parse_args()

    # Set conditional arguments to their defaults, if the conditions requiring them are satisfied and if they are not already provided
    for carg in conditional_args:
        if carg not in args:
            conditional_arg_default(carg, args, set_inplace=True)

    return parser, args, conditional_args


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
    parser, args, cargs = parse_args()

    # handle required model level hyperparams and non-default-valued params with for creating run (directory) name
    runname_arg_list = []
    for arg_name, arg_value in vars(args).items():
        if arg_name in ["alg", "backbone", "optim"]: # handle model arguments
            runname_arg_list.append(str(arg_value))
            continue
        if arg_name in cargs: # handle non-default conditional arguments
            if arg_value != conditional_arg_default(arg_name, args, set_inplace=False):
                runname_arg_list.append(f'{arg_name}={arg_value}')
            continue
        if (parser.get_default(arg_name) is not None) and (arg_value != parser.get_default(arg_name)):
            runname_arg_list.append(f'{arg_name}={arg_value}') # handle all other non-default args

    # create list of 'essential arguments' - required task relevant hparams (besides label), required model hparams, any non-default arguments, and any optional arguments, if provided. for data visualization
    essential_args = []
    essential_args.extend([args.nat, args.aug])
    essential_args.extend(runname_arg_list)
    
    # handle save directory and saved file name
    # runs are organized in a hierarchy: first by learning task settings (natural, augmentation, labelling schemes), then at the by model hyperparameters (alg, backbone, optim, emb_d etc.). each run has its own directory.
    if args.save_dir is not None: # save_dir refers to the directory to contain the directory for the run (run_dir)
        if not os.path.isdir(args.save_dir):
            os.makedirs(args.save_dir)
        save_dir = args.save_dir
    else: # default is ./outputs
        save_dir = './outputs'
    # save to folder specified by natural, augmentation, label settings
    run_dir = os.path.join(save_dir, "_".join([str(args.nat), str(args.aug), str(args.label)]))
    # if not os.path.isdir(save_dir):
    #     os.makedirs(save_dir)
    # run name without "_run#" appended
    runname_prefix = "_".join(runname_arg_list)
    # append "_run#" at end (start with 1 for first run)
    run_path = generate_runpath(os.path.join(run_dir, runname_prefix), 1)
    os.makedirs(run_path)

    # forbid inconsistent data generating parameters
    if args.aug == 'mult' and args.label != 'weights':
        raise Exception("'mult' augmentation scheme must be used with 'weights' labeling")
    # generate data
    tau_max = None
    if args.aug == 'corr':
        tau_max = args.tau_max
    data_dict = generate_cube_data(args.n, args.v, args.d, args.k, args.nat, args.aug, args.label, tau_max)
    (x1, x2, y), (val_x, val_y) = data_dict['train'], data_dict['val']

    # create dataloader
    trainset, valset = TensorDataset(x1, x2), TensorDataset(val_x, val_y)
    trainloader = DataLoader(trainset, batch_size=args.bs)
    valloader = DataLoader(valset, batch_size=args.bs)

    print(f'Train Loader length: {len(trainloader)}, Val Loader length: {len(valloader)}')

    # define embedding dimension
    emb_d = args.emb_dim

    # initialize backbone, using embedding dimension determined above
    if args.backbone == "linear":          
        backbone = nn.Linear(args.d, emb_d)
    elif args.backbone == "mlp":
        backbone = nn.Sequential(
            nn.Linear(args.d, 2*args.d),
            nn.ReLU(inplace=True),
            nn.Linear(2*args.d, emb_d)
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
        opt = optim.SGD(ssl_model.parameters(), lr=args.lr, weight_decay=args.wd)
    elif args.optim == "adam":
        opt = optim.Adam(ssl_model.parameters(), lr=args.lr, weight_decay=args.wd)
    else:
        raise Exception("Invalid argument 'optim', must be one of {'sgd', 'adam'}")
    
    epochs = args.epochs

    # linear classification loss on the validation set
    val_accs = []
    train_losses = []

    # Training loop. See ssl_model class for specific forward passes
    for epoch in range(epochs):
        print(f"Epochs {epoch}")
        start_time = time.time()

        # # debugging printing (part 1)
        # print("PRIOR TO EPOCH TRAINING")
        # print("Online backbone")
        # for name, param in ssl_model.online_backbone.named_parameters():
        #     print(name, param.requires_grad)
        # print("Target backbone")
        # for name, param in ssl_model.target_backbone.named_parameters():
        #     print(name, param.requires_grad)

        # UPDATE WEIGHTS OF ONLINE NETWORK
        ssl_model.train()
        # listen to gradient calculations for online network
        for param in ssl_model.online.parameters():
            param.requires_grad = True

        # print("INSIDE EPOCH, BEFORE LOOP")
        # print("Online backbone")
        # for name, param in ssl_model.online_backbone.named_parameters():
        #     print(name, param.requires_grad)
        # print("Target backbone")
        # for name, param in ssl_model.target_backbone.named_parameters():
        #     print(name, param.requires_grad)

        for idx, (x1, x2) in enumerate(trainloader):
            x1, x2 = x1.to(device), x2.to(device)
            opt.zero_grad()
            # run forward() on ssl_model, which is expected to return a SSL loss on x1, x2
            # the forward function is specific to each SSL algorithm (Spectral, MatrixSSL, etc.)
            loss = ssl_model(x1, x2)['loss']

            # store training loss with each training iteration
            train_losses.append(loss.item())

            # backprop
            loss.backward()
            del loss, x1, x2
            # update weights
            opt.step()
            
            # if idx == len(trainloader) - 2:
            #     for name, param in ssl_model.named_parameters():
            #         print(name, param.grad)

            # Momentum averaging for mssl with asymmetric networks
            if getattr(ssl_model, "asym", False):
                with torch.no_grad():
                    # terminates at end of shortest iterator, excludes predictor weights
                    # need to ensure backbone and projector networks the same   
                    for online_param, target_param in zip(ssl_model.online.parameters(), ssl_model.target.parameters()):
                        target_param.mul_(args.momentum).add_((1-args.momentum) * online_param)
        
        # every 20 epochs, save model weights. include time of save.
        if (epoch + 1) % 20 == 0:
            current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            torch.save(ssl_model.online_backbone.state_dict(), os.path.join(run_path, f"model_weights_{current_time}"))
        
    
        # # debugging printing (part )
        # print("POST EPOCH TRAINING")
        # print("Online backbone")
        # for name, param in ssl_model.online_backbone.named_parameters():
        #     print(name, param.requires_grad)
        # print("Target backbone")
        # for name, param in ssl_model.target_backbone.named_parameters():
        #     print(name, param.requires_grad)

        # EVALUATE DOWNSTREAM LIN CLF PERFORMANCE AT END OF EACH EPOCH
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
        "data": data_dict, # train, validation data
        "optim": args.optim,
        "args": args,
        "essential_args": essential_args,
        "val_accs":val_accs,
        "train_losses":train_losses
    }
    torch.save(run_dict, os.path.join(run_path, 'run_dict'))


if __name__ == '__main__':
    main()

# %%
