import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
from torch.distributions import MultivariateNormal

# split evaluate_embeddings into fitting linear classifier, and getting test scores

def fit_classifiers(dirs, val, val_labels, plot_train=False):
    """
    Fit linear classifier on each of representation functions in dirs, using validation data. 
    plot_train specifies whether to plot all training loss curves over fitting of classifiers
    Returns: (backbones, classifiers)
        backbones: list of backbones/representation functions, one per directory
        classifiers: list of trained nn.Linear modules, one per directory
    
    Requires val_labels to be {0, 1} labels rather than {-1, 1} labels, since uses BCE(WithLogits)Loss

    `dirs` can be a single directory, or a list of directories. If it is a single directory, all subdirectories are assumed to be runs.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # prepare datasets
    val_dataset = TensorDataset(val, val_labels)
    val_loader = DataLoader(val_dataset, batch_size=128)

    backbones = []
    classifiers = []
    val_losses_list = []

    if isinstance(dirs, str) and os.path.isdir(dirs):
        subdirs = [os.path.join(dirs, d) for d in os.listdir(dirs)]
        dirs = subdirs
    for idx, dir in enumerate(dirs):
        # print(f'Run {idx+1}')
        # load embedding function (backbone), create linear classifier (linear)
        run_dict = torch.load(os.path.join(dir, 'run_dict'))
        weights = run_dict['model_weights']
        key = 'weight' if 'weight' in weights.keys() else '0.weight'
        in_dim = weights[key].shape[1]
        emb_dim = run_dict['args'].emb_dim
        backbone_type = run_dict['args'].backbone
        if backbone_type == 'linear':
            backbone = nn.Linear(in_dim, emb_dim).to(device)
        elif backbone_type == 'relu':
            backbone = nn.Sequential(
                nn.Linear(in_dim, emb_dim),
                nn.ReLU(inplace=True)
            )   
        elif backbone_type == "mlp":
            backbone = nn.Sequential(
                nn.Linear(in_dim, 2*in_dim),
                nn.ReLU(inplace=True),
                nn.Linear(2*in_dim, emb_dim)
                )
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

def evaluate_normal_downstream(dirs, n_val, n_test, gt_idx=0, plot_train=False, print_stats=False):
    """
    For all runs in `dirs`, evaluate the downstream performance of the learned representations by training a binary linear classifier on validation data, and testing classification accuracy on test data.

    Natural/unagumented test/validation data is assumed to be distributed according to an isotropic normal with dimensions equal to the number of features * dimension of each feature. The label for test and validation is determined by a single feature indicated by `gt_idx`, which indexes the feature in gt_vecs list from the run's saved output.

    `dirs` can be a single directory, or a list of directories. If it is a single directory, all subdirectories are assumed to be runs.
    `n_val`, `n_test` indicate how many points of validation/test data in the procedure described above.
    """
    test_scores = [] # downstream/test classification accuracy score for linear(emb_dim, 1) classifier fit onto validation data
    if isinstance(dirs, str) and os.path.isdir(dirs):
        subdirs = [os.path.join(dirs, d) for d in os.listdir(dirs)]
        dirs = subdirs
    for idx, dir in enumerate(dirs):
        run_dict = torch.load(os.path.join(dir, 'run_dict'))
        gt_vecs = run_dict['data']['gt_vecs']
        d = gt_vecs[0].shape[0] # dimension of single feature
        K = len(gt_vecs) # number of features
        std_mvn = MultivariateNormal(torch.zeros(K*d), torch.eye(K*d))
        # have label of any datum depend only on the gt feature given by gt_idx
        label_vec = torch.zeros(K*d)
        label_vec[gt_idx*d:(gt_idx+1)*d] = gt_vecs[gt_idx]
        # Print true feature (that determines labels)
        if print_stats:
            print(f"Run {idx+1} label feature: {gt_vecs[gt_idx]}")
        # Print weights corresponding to the labeling feature that have been learnt (to see if they are similar/well learnt)
        # key = 'weight' if 'weight' in run_dict['model_weights'].keys() else '0.weight'
        # weights = run_dict['model_weights'][key]
        # print(f"Run {idx+1} feature weights: {weights[gt_idx, gt_idx*d:(gt_idx+1)*d]}")
        val_data = std_mvn.sample((n_val,))
        val_y = (torch.sign(val_data @ label_vec) + 1)/2
        test_data = std_mvn.sample((n_test,))
        test_y = (torch.sign(test_data @ label_vec) + 1)/2
        backbones, classifiers = fit_classifiers([dir], val_data, val_y, plot_train=plot_train)
        _, avg_acc = evaluate_classifiers(backbones, classifiers, test_data, test_y, print_stats=print_stats)
        test_scores.append(avg_acc) # since only one dir used
    avg_score = sum(test_scores) / len(test_scores)
    return test_scores, avg_score

def get_feature_downstream_scores(dirs, plot_train=False, print_stats=False):
    """
    Calls evaluate_normal_downstream for all possible features that define the label, for all directories in `dirs`. Assumes all runs in dirs are run on data with the same number of features.

    Returns an array of shape (len(dirs), num_features), where the (i, j)th entry denotes the downstream accuracy for run i, for downstream task based on feature j. Rows represent runs and columns represent features. A good representation should (at least) get good scores across the entire row.

    `dirs` can be a single directory, or a list of directories. If it is a single directory, all subdirectories are assumed to be runs.
    """
    if isinstance(dirs, str) and os.path.isdir(dirs):
        subdirs = [os.path.join(dirs, d) for d in os.listdir(dirs)]
        dirs = subdirs
    run_dict = torch.load(os.path.join(dirs[0], 'run_dict'))
    num_feats = run_dict['args'].num_feats
    scores_array = np.zeros((len(dirs), num_feats)) # one row per run, one col per feature
    for i in range(num_feats):
        print(f"Feature {i+1}")
        test_scores, _ = evaluate_normal_downstream(dirs, n_val=10000, n_test=10000, gt_idx=i, plot_train=plot_train, print_stats=print_stats)
        test_scores_arr = np.array([score.item() for score in test_scores])
        scores_array[:, i] = test_scores_arr
    return scores_array