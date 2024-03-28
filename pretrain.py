# set cwd set to matrixssl-inductive
import os
os.chdir(os.path.dirname(__file__))
from data.loader import generate_cube_data

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# def parse_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("task", help="specifies synthetic data generating process, one of 'cube_single', ")
#     parser.add_argument("backbone", help="architecture used, one of 'linear', ")
#     parser.add_argument("-n", help="number of datum", type=int)
#     parser.add_argument("-d", help="dimensionality of data", type=int)
#     parser.add_argument("-k", help="number invariant dimensions", type=int)
#     parser.add_argument("--split", help="specifies train-val split; input some number < n", type=int)
#     # might need conditional arguments... ?
#     args = parser.parse_args()
#     return args


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
    (x1, x2, _), (val_x, val_y) = generate_cube_data()

    # create dataloader
    trainset, valset = TensorDataset(x1, x2), TensorDataset(val_x, val_y)
    trainloader = DataLoader(trainset, batch_size=32)
    valloader = DataLoader(valset, batch_size=32)

    # ssl_model refers to both the encoder and the SSL class used to train it
    # the specific class of ssl_model is an SSL algorithm wrapper for the encoder
    ssl_model = None
    optim = None
    epochs = 100

    # Training loop. See ssl_model class for specific forward passes
    for epoch in range(epochs):

        # UPDATE WEIGHTS OF ENCODER
        ssl_model.train()
        # listen to gradient calculations for encoder
        for param in ssl_model.encoder.parameters():
            param.requires_grad = True

        for idx, (x1, x2) in enumerate(trainloader):
            optim.zero_grad()
            # run forward() on ssl_model, which is expected to return a SSL loss on x1, x2
            # the forward function is specific to each SSL algorithm (Spectral, MatrixSSL, etc.)
            loss = ssl_model(x1, x2)['loss']
            # backprop
            loss.backward()
            # update weights
            optim.step()
        
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






    pass


if __name__ == '__main__':
    main()
