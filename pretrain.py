# %%
# set cwd set to matrixssl-inductive
import os
os.chdir(os.path.dirname(__file__))
from data.loader import generate_cube_data
# %%
import argparse
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import TensorDataset, DataLoader

from models import Spectral, MatrixSSL

# %%

def parse_args():
    parser = argparse.ArgumentParser()

    # Algorithm, encoder, optimizer
    parser.add_argument("--alg", type=str, required=True, 
    help="SSL algorithm to train, one of {'spectral', 'mssl_a', 'mssl_s'}. mssl_a, mssl_s refer to MatrixSSL with asymmetric (online, target) networks and a single network, respectively")
    parser.add_argument("--backbone", type=str, required=True, help="architecture used, one of {'linear', 'mlp'}")
    parser.add_argument("--optim", type=str, required=True, help="optimizer, one of {'sgd', 'adam', 'adam_wd} adam_wd means adam with weight decay")

    # Data generation arguments
    parser.add_argument("--augmentation", type=str, required=True, help="specifies how positive pairs are generated, one of {'mult', 'add'}; also determines labeling function. see generate_cube_data function in ./data/loader.py for more information")

    parser.add_argument("-n", type=int, default=(2 ** 16) + 12500, help="number of data points")
    parser.add_argument("-v", type=int, default=12500, help="size of validation set, input some v<n")
    parser.add_argument("-d", type=int, default=50, help="dimension of data")
    parser.add_argument("-k", type=int, default=10, help="number invariant dimensions")

    # Other training arguments
    parser.add_argument("--epochs", type=int, default=150, help="number training epochs")
                        # TODO Change default here ^ back to 500 <- 150 once done experimenting
                        # TODO Change default here v back to 512... ?
    parser.add_argument("--bs", type=int, default=128, help="training minibatch size")
    parser.add_argument("--lr", type=float, default=1e-6, help="optimizer learning rate hyperparam")
    parser.add_argument("--wd", type=float, default=1e-6, help="optimizer weight decay hyperparam")
    # weight decay values tried for adam: 1e-6, 5e-6

    # Conditional arguments 
    parser.add_argument("--momentum", type=float, help="momentum averaging parameter for MatrixSSL with asymmetric networks. required if alg set to mssl_a, must be in [0, 1]")
    # parser.add_argument("--hidden_dim", type=int, default=20, help="dimension of hidden layer in predictor network for MatrixSSL with assymetric networks. required if alg set to mssl_a")

    # Saving/loading
    parser.add_argument("--save_dir", type=str, required=False, help="directory to save model weights/run details to")

    args = parser.parse_args()
    return args


def fit_classifier(linear, backbone, valloader, clf_loss_fn, clf_optim, clf_epochs=None):
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
        # zero gradients
        clf_optim.zero_grad()
        # get linear classif predictor
        pred = torch.sigmoid(linear(backbone(x))).flatten()
        # calculate loss on labels, backprop
        clf_loss = clf_loss_fn(pred, y)
        clf_loss.backward()
        # update weights
        clf_optim.step()
    
    with torch.no_grad():
        total_count = 0
        correct_count = 0
        for idx, (x, y) in enumerate(valloader):
            pred = (torch.sigmoid(linear(backbone(x))) >= 0.5).flatten()
            correct_count += torch.sum(pred == y)
            total_count += len(y)
        clf_acc = correct_count / total_count
    return clf_acc

def generate_filepath(save_path:str, num):
    """
    Given directory path to save to, returns a compatible path name that doesn't override existing run log files

    save_path: path to save file to, without "_run#" appended
    returns: save_path with compatible "_run#" string appended
    """
    candidate_path = save_path + "_run" + str(num)
    if os.path.isfile(candidate_path):
        num += 1
        return generate_filepath(save_path, num)
    else:
        return candidate_path


def main():
    args = parse_args()
    
    if args.save_dir is not None:
        save_dir = args.save_dir if os.path.isdir(args.save_dir) else os.makdirs(args.save_dir)
    else: # save to default: outputs folder
        save_dir = './outputs'
    # file name without "_run#" appended
    filename_prefix = "_".join([f'{args.alg}', f'{args.backbone}', f'{args.optim}', f'augment={args.augmentation}', f'epochs={args.epochs}', f'bs={args.bs}', f'wd={args.wd}'])
    # append "_run#" at end (start with 1 for first file)
    save_path = generate_filepath(os.path.join(save_dir, filename_prefix), 1)

    # generate data
    data_dict = generate_cube_data(args.n, args.v, args.d, args.k, args.augmentation)
    (x1, x2, y), (val_x, val_y) = data_dict['train'], data_dict['val']

    # create dataloader
    trainset, valset = TensorDataset(x1, x2), TensorDataset(val_x, val_y)
    trainloader = DataLoader(trainset, batch_size=args.bs)
    valloader = DataLoader(valset, batch_size=args.bs)

    print(f'Train Loader length: {len(trainloader)}, Val Loader length: {len(valloader)}')

    # initialize backbone
    if args.backbone == "linear":
        backbone = nn.Linear(args.d, args.k)
    elif args.backbone == "mlp":
        backbone = nn.Sequential(
            nn.Linear(args.d, 2*args.d),
            nn.ReLU(inplace=True),
            nn.Linear(2*args.d, args.k)
            )
    else:
        raise Exception("Invalid argument 'backbone', must be one of {'linear', 'mlp'}")

    # ssl_model refers to both the backbone and the SSL class used to train it
    # the specific class of ssl_model is an SSL algorithm wrapper for the backbone
    if args.alg == "spectral":
        ssl_model = Spectral(backbone=backbone, emb_dim=args.k)
    elif args.alg == "mssl_a":
        if args.momentum is None or (not (0 <= args.momentum <= 1)):
            raise Exception("Momentum averaging parameter must be set to a value within [0, 1] for asymmetric MatrixSSL")
        ssl_model = MatrixSSL(backbone=backbone, emb_dim=args.k, asym=True, momentum=args.momentum)
    elif args.alg == "mssl_s":
        ssl_model = MatrixSSL(backbone=backbone, emb_dim=args.k, asym=False)
    else:
        raise Exception("Invalid argument 'alg', must be one of {'spectral', 'mssl_a', 'mssl_s'}")

    if args.optim == "sgd":
        opt = optim.SGD(ssl_model.parameters(), lr=1e-3, weight_decay=args.wd)
    elif args.optim == "adam":
        opt = optim.Adam(ssl_model.parameters(), lr=1e-3, weight_decay=args.wd)
        # values tried for wd: 1e-6,
    else:
        raise Exception("Invalid argument 'optim', must be one of {'sgd', 'adam'}")
    
    epochs = args.epochs

    # linear classification loss on the validation set (not sure if this is supposed to be called a validation set)
    val_accs = []
    train_losses = []

    # Training loop. See ssl_model class for specific forward passes
    for epoch in range(epochs):

        # UPDATE WEIGHTS OF ONLINE NETWORK
        ssl_model.train()
        # listen to gradient calculations for online network
        for param in ssl_model.online.parameters():
            param.requires_grad = True

        for idx, (x1, x2) in enumerate(trainloader):
            opt.zero_grad()
            # run forward() on ssl_model, which is expected to return a SSL loss on x1, x2
            # the forward function is specific to each SSL algorithm (Spectral, MatrixSSL, etc.)
            loss = ssl_model(x1, x2)['loss']

            # store training loss with each training iteration
            train_losses.append(loss.item())

            # backprop
            loss.backward()
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
        
        # EVALUATE DOWNSTREAM LIN CLF PERFORMANCE AT END OF EACH EPOCH
        ssl_model.eval()
        # stop calculating gradients for backbone in online network
        for param in ssl_model.online_backbone.parameters():
            param.requires_grad = False

        # instantiate linear classifier
        linear = nn.Linear(ssl_model.emb_dim, 1)
        # fit linear classifier, print classification accuracy on validation set
        clf_acc = fit_classifier(
            linear,
            ssl_model.online_backbone,
            valloader,
            nn.BCELoss(),
            optim.SGD(linear.parameters(), lr=0.1, momentum=0.9)
            )
        print(f'Epoch {epoch+1} classification accuracy: {clf_acc}')
        val_accs.append(clf_acc)

    # store model weights, trainval+true_w data, batch_size (for recreating dataloader), optimizer to save_path 
    run_dict = {
        "model_weights":ssl_model.online_backbone.state_dict(),
        "data": data_dict, # train, validation data
        "optim": args.optim,
        "args": args,
        "val_accs":val_accs,
        "train_losses":train_losses
    }
    torch.save(run_dict, save_path)


if __name__ == '__main__':
    main()

# %%
