# %%
import os
# os.chdir('..')
import math
import torch
import numpy as np
import matplotlib.pyplot as plt
# %%

def plot_metrics(dirs, toplot, aggregate=True, plotall=False):
    """
    Plots specified metric of a single run or of multiple runs. 
    Each directory in dirs expects a path to the exact run directory.
    """
    metrics_list = []
    for dir in dirs:
        run_dict = torch.load(os.path.join(dir, 'run_dict'))
        metrics = run_dict[toplot]
        metrics_list.append(metrics)
    metrics_np = np.asarray(metrics_list)
    plt.figure()
    if aggregate:
        avgs = metrics_np.mean(axis=0)
        stds = metrics_np.std(axis=0)
        x = [i for i in range(len(avgs))]
        plt.plot(x, avgs, 'k', label='average')
        plt.fill_between(x, avgs-stds, avgs+stds,
            alpha=0.2, facecolor='#089FFF',
            linewidth=4, antialiased=True)
    if plotall:
        for series, run_name in zip(metrics_list, dirs):
            plt.plot(series, label=run_name)



def filter_and_plot(filterlist, to_plot, dir='./outputs', aggregate=True, plotall=False, **kwargs):
    """
    Filters from files in a directory, and plots their metrics in a single axes
    """
    dicts = []
    filenames = []
    for file in os.listdir(dir):
        if all(word in file for word in filterlist):
            d = torch.load(os.path.join(dir, file))
            dicts.append(d)
            filenames.append(file)
    values = [d[to_plot] for d in dicts]
    
    plt.figure()
    if aggregate:
        values_np = np.asarray(values)
        avgs = values_np.mean(axis=0)
        stds = values_np.std(axis=0)
        x = [i for i in range(len(avgs))]
        plt.figure()
        plt.plot(x, avgs, 'k', label='average')
        plt.fill_between(x, avgs-stds, avgs+stds,
            alpha=0.2, facecolor='#089FFF',
            linewidth=4, antialiased=True)
    
    if plotall:
        for series, filename in zip(values, filenames):
            plt.plot(series, label=filename)

    if to_plot == 'val_accs':
        plt.ylim(0.5, 1)
        plt.ylabel('Classification Accuracy')
        plt.xlabel('Epochs')
    elif to_plot == 'train_losses':
        plt.ylabel('Train Loss')
        plt.xlabel('Train Steps')

    for key, val in kwargs.items():
        if key == 'title':
            plt.title(val)

    plt.legend(bbox_to_anchor=[1, 0.5], loc='center left')
    plt.show()
            

def show_weights(names, single_dir=None, plot_weights=True, subtitle=True, title=True):
    """
    Plot weight matrices across runs (single_dir=None), or weight matrices within the same run (single_dir not None)

    single_dir determines if names is interpreted as a list of directories/runs for which we plot all final weight matrices in run_dict (use case: same parameters but over multiple runs, so single_dir=None), or a list of filenames from which we plot (use case: weights over time in a single run, so then dir must be provided).

    when needed, dir expects the relative path to the exact run directory
    """
    subplot_width = 2.5
    subplot_height = 2.5
    if plot_weights:
        n_plots = len(names)
        n_cols = 3
        n_rows = math.ceil(n_plots / n_cols)
        if n_plots == 1:
            fig, ax = plt.subplots(1, figsize=(subplot_width, subplot_height), constrained_layout=True)
        else:
            fig, ax = plt.subplots(n_rows, n_cols, figsize=(subplot_width*n_cols, subplot_height*n_rows), constrained_layout=True)
        if single_dir is None: # we are plotting final weights for one or more runs/directories
            for idx, run_dir in enumerate(names):
                row = idx // 3
                col = idx % 3
                run_dict = torch.load(os.path.join(run_dir, 'run_dict'))
                weights = run_dict['model_weights']
                current_ax = ax if n_plots == 1 else ax[row, col]
                im = current_ax.imshow(weights['weight'], cmap='hot', interpolation='nearest', vmin=-0.8, vmax=0.8)
                if subtitle:
                    current_ax.set_title(f"run {idx+1}") # intended to be used for multiple runs within the same params
        else: # we are plotting weights over time for a single run/directory
            assert single_dir is not None
            for idx, filename in enumerate(names):
                # row = idx // 3 # uncomment this soon
                # col = idx % 3
                weights = torch.load(os.path.join(single_dir, filename))
                current_ax = ax if n_plots == 1 else ax[idx]
                im = current_ax.imshow(weights['weight'], cmap='hot', interpolation='nearest', vmin=-0.8, vmax=0.8)
                # ax[idx].title(f"run {idx+1}")
            run_dict = torch.load(os.path.join(single_dir, 'run_dict'))
        if title == True:
            fig.suptitle(f"{' '.join(run_dict['essential_args'])}")
        elif title:
            fig.suptitle(title)
        # plt.subplots_adjust(hspace=0.1)
        # add colorbar
        cbar_ax = ax if n_plots == 1 else ax[:, :]
        fig.colorbar(im, ax=cbar_ax, location='bottom', orientation='horizontal')
        # Adjust the spacing between subplots to make room for the colorbar
        # fig.subplots_adjust(bottom=0.15)
        for index in range(n_plots, n_rows * n_cols):
            row = index // n_cols
            col = index % n_cols
            fig.delaxes(ax[row, col])
        plt.show()
    

def plot_embeddings(dirs, toplot, diff=False, n=20, print_stats=True, subtitle=False):
    """
    selects a random subset of data and plots the embeddings or the covariance of embeddings

    toplot = embds plots embedding matrices Z1, Z2 (row = data)
    toplot = cov plots (Z1-mu1)(Z1-mu1).T and (Z2-mu2).T(Z2-mu2)

    n: specifies number of embeddings to use when calculating embedding difference stats or to show when toplot = embds

    diff: determines if we plot both matrices in toplot, or the difference between the two
    """
    n_plots = len(dirs)
    subplot_width = 2
    subplot_height=2.5
    if diff:
        if n_plots == 1:
            fig, ax = plt.subplots(figsize=(subplot_width, subplot_height))
        else:
            fig, ax = plt.subplots(1, n_plots, figsize=(n_plots*subplot_width, subplot_height))
    else:
        fig, ax = plt.subplots(n_plots, 2, figsize=(2*subplot_width, n_plots*subplot_height))

    weight_list = []
    x1_list = []
    x2_list = []
    embd_diff_norm_list = []
    cov_diff_norm_list = []
    aug_contribution_list = [] # list of squares of contribution ratios of augmented dimensions of embedding differences, to measure contribution of difference elements corresp to augmented dims. 1 - this gives corresp. contribution of unaugmented dims
    for idx, dir in enumerate(dirs):
        run_dict = torch.load(os.path.join(dir, 'run_dict'))
        weights = run_dict['model_weights']['weight']
        x1, x2 = run_dict['data']['train'][0], run_dict['data']['train'][1]
        k = run_dict['args'].k # number of unaugmented dims, assumed all the same across all runs

        weight_list.append(weights)
        x1_list.append(x1)
        x2_list.append(x2)

    for idx, (x1, x2, weights) in enumerate(zip(x1_list, x2_list, weight_list)):
        # calculate embedding differences
        indices = torch.randperm(x1.shape[0])[:n]
        subset1, subset2 = x1[indices], x2[indices]
        embeds1, embeds2 = subset1 @ weights.T, subset2 @ weights.T # direct embedding matrices, embeds are rows
        if idx == 0:
            print(f"Norm of {n} rows of Z1 for run {idx+1}: {torch.norm(embeds1, dim=1)}")
        embed_diffs = embeds1 - embeds2
        # calculate norms of differences in embeddings
        norm_diffs = torch.norm(embed_diffs, dim=1)
        embd_diff_norm_list.append(torch.mean(norm_diffs))

        # calculate (normalized) contribution of augmented dimensions to norm
        aug_dim_sq_norm = torch.norm(embed_diffs[:, k:], dim=1) ** 2
        normalized_contrib = aug_dim_sq_norm / (norm_diffs ** 2)
        # print(f"Normalized contributions of aug dimensions in {n} embeddings for run {idx+1}: {normalized_contrib}")
        aug_contribution_list.append(torch.mean(normalized_contrib))

        # calculate covariance differences
        z1, z2 = x1 @ weights.T, x2 @ weights.T
        mean1, mean2 = z1.mean(dim=0), z2.mean(dim=0)
        centered1, centered2 = z1-mean1, z2-mean2
        B = centered1.shape[0]
        cov1 = centered1.T @ centered1 / (B-1)  # sample covariance matrices
        cov2 = centered2.T @ centered2 / (B-1)
        cov_diff = cov1 - cov2
        # calculate frobenius norm of differences in covariances
        frob_norm = torch.norm(cov_diff)
        cov_diff_norm_list.append(frob_norm) 

        figtitle = 'Embedding' if toplot == 'embds' else 'Covariance'

        if n_plots == 1:
            current_ax = ax
        else:
            current_ax = ax[idx]
        if diff: # plot difference of matrices across the two sets of augmentations
            # select matrix to plot
            if toplot == 'embds':
                matrix_toplot = embed_diffs
            elif toplot == 'cov':
                matrix_toplot = cov_diff
            # plot matrix
            im = current_ax.imshow(matrix_toplot, cmap='hot', interpolation='nearest')
            # show subtitles for each run
            if subtitle:
                current_ax.set_title(f'Run {idx+1}')
            # add more figtitle details
            figtitle += ' Differences'
            if toplot == 'embds':
                figtitle += f', {n} points'

        else: # plot matrix for each set of augmentations
            # select matrices to plot
            if toplot == 'embds':
                matrix_toplot1 = embeds1
                matrix_toplot2 = embeds2
            elif toplot == 'cov':
                matrix_toplot1 = cov1
                matrix_toplot2 = cov2
            # plot matrices
            if n_plots == 1: # ax.shape = (2,)
                current_ax[0].imshow(matrix_toplot1, cmap='hot', interpolation='nearest')
                im = current_ax[1].imshow(matrix_toplot2, cmap='hot', interpolation='nearest')
            else: # ax.shape = (n_plots, 2), for n_plots > 1
                current_ax[0].imshow(matrix_toplot1, cmap='hot', interpolation='nearest')
                im = current_ax[1].imshow(matrix_toplot2, cmap='hot', interpolation='nearest')
                current_ax[0].set_title(f'Run {idx + 1}')
    
    if print_stats:
        if toplot == 'embds':
            norm_avg = sum(embd_diff_norm_list) / len(embd_diff_norm_list)
            print(f'Average norm of differences in embeddings ({n_plots} runs): {norm_avg}, embed dim = {embed_diffs.shape[1]}')
        elif toplot == 'cov':
            norm_avg = sum(cov_diff_norm_list) / len(cov_diff_norm_list)
            print(f'Average Frobenius norm of differences in covariances ({n_plots} runs): {norm_avg}, embed dim = {cov_diff.shape[1]}')
        print(f"Average normalized contribution to norm of augmented dimensions for each run: {aug_contribution_list}")

    cbar_ax = ax if n_plots == 1 else ax[:]
    fig.colorbar(im, ax=cbar_ax, location='bottom', orientation='horizontal')
    fig.suptitle(f"{figtitle}, {' '.join(run_dict['essential_args'])}")
    plt.show()


# %%
# def plot_assumptions(*dirs, type, subtitle=True, title=True):
#     n_plots = len(dirs)
#     fig, ax = plt.subplots(n_plots, figsize=(3, 1.75*n_plots), constrained_layout=True)
#     for idx, run_dir in enumerate(dirs):
#         run_dict = torch.load(os.path.join(run_dir, 'run_dict'))
#         weights = run_dict['model_weights']
#         ##
#         im = ax[idx].imshow(weights['weight'], cmap='hot', interpolation='nearest', vmin=-0.8, vmax=0.8)
#         if subtitle:
#             ax[idx].set_title(f"{' '.join(run_dict['essential_args'])} run {idx}")

def plot_assumptions(weights, data):
    """
    Plot 
    """
    i = torch.randint(low=0, high=data.shape[0], size=(1,)).item()
    x = data[i]
    x_pairs = generate_aug_from_bool_aug(x, 100, k)
    # some sort of check to see if mean of x_pairs is equal to x, and same for w^T x = w^T E[x_pairs]...
    # hypothesis testing...?
    mean_x_pairs = x_pairs.mean(dim=0)
    wx = weights @ x
    wEx = weights @ mean_x_pairs

    plt.figure()
    im1= plt.imshow(wx.unsqueeze(0), cmap='hot')
    plt.title('$Wx$')
    plt.colorbar(im1)
    plt.figure()
    im2 = plt.imshow(wEx.unsqueeze(0), cmap='hot')
    plt.title("$WE[x'|x]$")
    plt.colorbar(im2)
    # print('Comparing data')
    # print(f'x: \n {x[k:]}')
    # print(f"E[x'|x]: \n {mean_x_pairs[k:]}")
    # print(f'Weights: {weights}')
    # print(f'Norm of difference: {torch.norm(x - mean_x_pairs)}')
    # print('Comparing output of classifier')
    # print(f'W x: \n {wx}')
    # print(f"WE[x'|x]: \n {wEx}")
    # print(f'Norm of difference: {torch.norm(wx - wEx)}')

    cov_x = torch.outer(x, x) # rank 1 matrix, calculated with only one vector
    cov_x_pairs = (x_pairs - mean_x_pairs).T @ (x_pairs - mean_x_pairs) / (x.shape[0])
    wxxw = weights @ cov_x @ weights.T
    wExxw = weights @ cov_x_pairs @ weights.T

    plt.figure()
    im3= plt.imshow(wxxw, cmap='hot')
    plt.title('$W xx^T W^T$')
    plt.colorbar(im3)
    plt.figure()
    im4 = plt.imshow(wExxw, cmap='hot')
    plt.title("$W E[x'x'^T |x] W^T$")
    plt.colorbar(im4)

    # # print('Comparing covariances')
    # # print(f'Norm of difference: {torch.norm(cov_x - cov_x_pairs)}')
    # # print('Comparing output of classifier')
    # print(f'W xx^T W^T: {wxxw[0]}')
    # print(f"W E[x'x'^T|x] W^T: {wExxw[0]}")
    # # print(f'Norm of difference: {torch.norm(wxxw - wExxw)}')