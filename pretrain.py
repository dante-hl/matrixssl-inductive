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

from models.backbones import Spectral, MatrixSSL

# %%

def parse_args():
    parser = argparse.ArgumentParser()
    # TODO set defaults for n,d,k,v,weights (?)
    parser.add_argument("--alg", type=str, help="SSL algorithm to train, one of {'spectral', 'mssl_a', 'mssl_s'}. mssl_a, mssl_s refer to MatrixSSL with assymetric (online, target) networks and a single network, respectively")
    parser.add_argument("--backbone", type=str, help="architecture used, one of {'linear', 'mlp'}")
    parser.add_argument("-n", type=int, default=50000, help="number of data points")
    parser.add_argument("-v", type=int, default=12500, help="size of validation set, input some v<n")
    parser.add_argument("-d", type=int, default=50, help="dimension of data")
    parser.add_argument("-k", type=int, default=10, help="number invariant dimensions")
    parser.add_argument("--epochs", type=int, default=500, help="number training epochs")
    parser.add_argument("--batch_size", type=int, default=512, help="size of training minibatch")
    parser.add_argument("--optim", type=str, help="optimizer, one of {'sgd', 'adam', 'adam_wd} adam_wd means adam with weight decay")
    parser.add_argument("--weights", help="weight tensor for true classification function, optional", nargs="*")
    parser.add_argument("--save_dir", type=str, required=False, help="directory to save model weights/run details to")

    args = parser.parse_args()
    return args


def fit_classifier(linear, encoder, valloader, clf_loss_fn, clf_optim, clf_epochs=None):
    """
    Fits linear classifier on representations learned so far using validation dataset, outputs classification accuracy of classifier after fitting. Returns classification accuracy on validation set

    Ensure encoder has requires_gradients=False (and encoder set to eval()) before running

    linear: the linear classifier to fit
    encoder: encoder/embedding function
    valloader: DataLoader for validation set to fit on
    clf_loss_fn: linear classification loss function
    clf_optim: optimizer for linear classifier
    clf_epochs: number of training epochs (currently ignored, only using 1)
    """

    for idx, (x, y) in enumerate(valloader):
        # zero gradients
        clf_optim.zero_grad()
        # get linear classif predictor
        pred = torch.sigmoid(linear(encoder(x)))
        # calculate loss on labels, backprop
        clf_loss = clf_loss_fn(pred, y)
        clf_loss.backward()
        # update weights
        clf_optim.step()
    
    with torch.no_grad():
        total_count = 0
        correct_count = 0
        for idx, (x, y) in enumerate(valloader):
            pred = (torch.sigmoid(linear(encoder(x))) >= 0.5)
            correct_count += torch.sum(pred == y)
            total_count += len(y)
        clf_acc = correct_count / total_count
    return clf_acc


# ideally: just provide args for model, then train
def main():
    args = parse_args()
    
    if args.save_dir is not None:
        save_dir = args.save_dir if os.path.isdir(args.save_dir) else os.makdirs(args.save_dir)
    else: # save to default
        save_dir = './outputs'
    filename = "_".join([f'{args.alg}', f'{args.backbone}', f'{args.optim}'])
    save_path = os.path.join(save_dir, filename)

    if len(args.weights) > 0:
        assert len(args.weights) != args.k
        weights = torch.tensor(args.weights, dtype=torch.float32)
    else:
        weights = None

    # generate data
    data_dict = generate_cube_data(args.n, args.v, args.d, args.k, weights)
    (x1, x2, _), (val_x, val_y) = data_dict['train'], data_dict['val']
    # create dataloader
    trainset, valset = TensorDataset(x1, x2), TensorDataset(val_x, val_y)
    trainloader = DataLoader(trainset, batch_size=args.batch_size)
    valloader = DataLoader(valset, batch_size=args.batch_size)

    # initialize encoder
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

    # ssl_model refers to both the encoder and the SSL class used to train it
    # the specific class of ssl_model is an SSL algorithm wrapper for the encoder
    if args.alg == "spectral":
        ssl_model = Spectral(backbone=backbone, emb_dim=args.k)
    elif args.alg == "mssl_a":
        ssl_model = MatrixSSL(backbone=backbone, emb_dim=args.k, assym=True)
    elif args.alg == "mssl_s":
        ssl_model = MatrixSSL(backbone=backbone, emb_dim=args.k, assym=False)
    else:
        raise Exception("Invalid argument 'alg', must be one of {'spectral', 'mssl_a', 'mssl_s'}")

    if args.optim == "sgd":
        opt = optim.SGD(ssl_model.parameters(), lr=0.01)
    elif args.optim == "adam":
        opt = optim.Adam(ssl_model.parameters(), lr=10e-3)
    elif args.optim == "adam_wd":
        opt = optim.Adam(ssl_model.parameters(), lr=10e-3, weight_decay=0.004)
    else:
        raise Exception("Invalid argument 'optim', must be one of {'sgd', 'adam', 'adam_wd'}")
    
    epochs = args.epochs

    # Training loop. See ssl_model class for specific forward passes
    for epoch in range(epochs):

        # UPDATE WEIGHTS OF ENCODER
        ssl_model.train()
        # listen to gradient calculations for encoder
        for param in ssl_model.encoder.parameters():
            param.requires_grad = True

        for idx, (x1, x2) in enumerate(trainloader):
            opt.zero_grad()
            # run forward() on ssl_model, which is expected to return a SSL loss on x1, x2
            # the forward function is specific to each SSL algorithm (Spectral, MatrixSSL, etc.)
            loss = ssl_model(x1, x2)['loss']
            # backprop
            loss.backward()
            # update weights
            opt.step()
        
        # EVALUATE DOWNSTREAM LIN CLF PERFORMANCE AT END OF EACH EPOCH
        ssl_model.eval()
        # stop calculating gradients for encoder
        for param in ssl_model.encoder.parameters():
            param.requires_grad = False

        # instantiate linear classifier
        linear = nn.Linear(ssl_model.emb_dim, 1)
        # fit linear classifier, print classification accuracy on validation set
        clf_acc = fit_classifier(
            linear,
            ssl_model.encoder,
            valloader,
            nn.BCELoss(),
            optim.SGD(linear.parameters(), lr=0.1, momentum=0.9)
            )
        print(f'Epoch {epoch+1} classification accuracy: {clf_acc}')

    # store model weights, trainval+true_w data, batch_size (for recreating dataloader), optimizer to save_path 
    run_dict = {
        "model_weights":ssl_model.backbone.state_dict(),
        "data": data_dict, # train, validation data
        "batch_size": args.batch_size,
        "optim": args.optim
    }
    torch.save(run_dict, save_path)


if __name__ == '__main__':
    main()
