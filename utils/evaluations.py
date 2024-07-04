import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np

# split evaluate_embeddings into fitting linear classifier, and getting test scores

def fit_classifiers(dirs, val, val_labels, plot_train=False):
    """
    Fit linear classifier on each of representation functions in dirs, using validation data. 
    plot_train specifies whether to plot all training loss curves over fitting of classifiers
    Returns: (backbones, classifiers)
        backbones: list of backbones/representation functions, one per directory
        classifiers: list of trained nn.Linear modules, one per directory
    
    Requires val_labels to be {0, 1} labels rather than {-1, 1} labels, since uses BCE(WithLogits)Loss
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # prepare datasets
    val_dataset = TensorDataset(val, val_labels)
    val_loader = DataLoader(val_dataset, batch_size=128)

    backbones = []
    classifiers = []
    val_losses_list = []
    for idx, dir in enumerate(dirs):
        # print(f'Run {idx+1}')
        # load embedding function (backbone), create linear classifier (linear)
        run_dict = torch.load(os.path.join(dir, 'run_dict'))
        weights = run_dict['model_weights']
        emb_dim = run_dict['args'].emb_dim
        backbone = nn.Linear(weights['weight'].shape[1], weights['weight'].shape[0]).to(device)
        backbone.load_state_dict(weights)
        # no weight updates for backbone
        for param in backbone.parameters():
            param.requires_grad = False
        linear = nn.Linear(emb_dim, 1)
        optimizer = optim.SGD(linear.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-5)
        loss_fn = nn.BCEWithLogitsLoss()
        # fit lin clf on validation data
        val_losses = []
        for epoch in range(50):
            for idx, (x, y) in enumerate(val_loader):
                x, y, = x.to(device), y.to(device)
                optimizer.zero_grad()
                logits = linear(backbone(x))
                clf_loss = loss_fn(logits.squeeze(), y)
                val_losses.append(clf_loss.item())
                clf_loss.backward()
                del x, y
                optimizer.step()
            # if epoch % 10 == 0:
            #     print(f"Loss at epoch {epoch}: {clf_loss}")
        backbones.append(backbone)
        classifiers.append(linear) 
        val_losses_list.append(val_losses)
    if plot_train:
        fig, ax = plt.subplots()
        val_losses_arr = np.array(val_losses_list)
        for i in range(val_losses_arr.shape[0]):
            ax.plot(val_losses_arr[i], label=f'run {i+1}')
        ax.set_ylabel('Mean (minibatch) BCE Loss')
        ax.set_xlabel('Train Steps')
        plt.legend(loc='upper right')
        plt.show()
    return backbones, classifiers 

def evaluate_classifiers(backbones, classifiers, test, test_labels, print_stats=False):
    """
    Evaluate each trained linear classifier in `classifiers` on test data.

    backbones: list of backbones/representation functions
    classifiers: list of trained nn.Linear modules
    Returns: (test_accs, avg_acc)
        test_accs: list of lin clf accuracies on test data for each linear classifier + backbone
        avg_accs: average of accs in test_accs

    Requires test_labels to be {0, 1} labels rather than {-1, 1} labels
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    test_dataset = TensorDataset(test, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=128)

    test_accs = []
    for idx, (backbone, linear) in enumerate(zip(backbones, classifiers)):
        # get test set accuracy
        with torch.no_grad():
            correct = 0
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                out = linear(backbone(x)).flatten()
                out[out == 0] = 1 # also included same 'convention' in labeling - not sure if this makes sense or not.
                pred = (torch.sign(out) + 1)/2
                correct += (pred == y).float().sum()
        test_acc = correct / len(test_dataset)
        test_accs.append(test_acc)
        if print_stats:
            print(f"Correct: {correct} out of {len(test_dataset)} = {test_acc}")
    
    avg_acc = sum(test_accs) / len(test_accs)
    return test_accs, avg_acc