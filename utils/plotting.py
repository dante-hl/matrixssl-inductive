# %%
import os
# os.chdir('..')
import math
import torch
import numpy as np
import matplotlib.pyplot as plt
# %%

def plot_metrics(dirs, toplot='train_losses', aggregate=True, plotall=False):
    """
    Plots specified metric of all runs in `dirs`. `dirs` can be a single directory, or a list of directories. If it is a single directory, all subdirectories are assumed to be runs.

    toplot specifies metric to be plotted. includes {'train_losses', 'gradient_norms'}
    """
    metrics_dict = {}
    title_append = None
    # create empty lists for each metric. these will contain series from each run.
    if toplot == 'train_losses':
        metrics_dict[toplot] = []
    elif toplot == 'gradient_norms':
        metrics_dict['weight'] = []
        metrics_dict['bias'] = []
        title_append = "gradient norms"
    # if dirs is a single directory create list of subdirectories
    if isinstance(dirs, str) and os.path.isdir(dirs): 
        subdirs = [os.path.join(dirs, d) for d in os.listdir(dirs)]
        dirs = subdirs
    # for the specified metric, add each run's series to its corresponding list
    for dir in dirs:
        run_dict = torch.load(os.path.join(dir, 'run_dict'))
        if toplot == 'train_losses':
            metrics_dict[toplot].append(run_dict[toplot])
        elif toplot == 'gradient_norms':
            metrics_dict['weight'].append(run_dict[toplot]['weight'])
            metrics_dict['bias'].append(run_dict[toplot]['bias'])
        # metrics = run_dict[toplot]
        # metrics_list.append(metrics)
    
    fig, ax = plt.subplots(1, len(metrics_dict))
    ax = np.atleast_1d(ax)
    for idx, key in enumerate(metrics_dict.keys()):
        curr_ax = ax[idx]
        metrics_lists = metrics_dict[key] # get list of series for specific metric
        metrics_np = np.asarray(metrics_lists) # make into np array (one row per series)
        if aggregate:
            avgs = metrics_np.mean(axis=0)
            stds = metrics_np.std(axis=0)
            x = [i for i in range(len(avgs))]
            curr_ax.plot(x, avgs, 'k', label='average')
            curr_ax.fill_between(x, avgs-stds, avgs+stds,
                alpha=0.2, facecolor='#089FFF',
                linewidth=4, antialiased=True)
        if plotall:
            for series, run_name in zip(metrics_lists, dirs):
                curr_ax.plot(series, label=run_name.split("_")[-1])
            plt.legend(loc='upper right')
        curr_ax.set_title(f"{key} {title_append}" if title_append is not None else key)
    plt.show()
            

def show_weights(dirs, over_time=False, plot_weights=True, subtitle=True, title=True):
    """
    Plot weight matrices across runs in `dirs` (over_time=False), or weight matrices across time within the same run (over_time=True).
    """
    subplot_width = 2.5
    subplot_height = 2.5
    if plot_weights:
        if over_time:
            # expect dirs to be single path string, assert all subpaths of dirs are not directories
            subpaths = [os.path.join(dirs, d) for d in os.listdir(dirs) if not d.endswith('run_dict')]
            assert all(not os.path.isdir(subpath) for subpath in subpaths)
            n_plots = len(subpaths) - 1
        else:
            if isinstance(dirs, str) and os.path.isdir(dirs):
                subdirs = [os.path.join(dirs, d) for d in os.listdir(dirs)]
                dirs = subdirs
            n_plots = len(dirs)
        n_cols = 3
        n_rows = math.ceil(n_plots / n_cols)
        if n_plots == 1:
            fig, ax = plt.subplots(1, figsize=(subplot_width, subplot_height), constrained_layout=True)
        else:
            fig, ax = plt.subplots(n_rows, n_cols, figsize=(subplot_width*n_cols, subplot_height*n_rows), constrained_layout=True)
            ax = ax.flatten()
        if not over_time: # plotting final weights for all runs in dirs
            if isinstance(dirs, str) and os.path.isdir(dirs):
                subdirs = [os.path.join(dirs, d) for d in os.listdir(dirs)]
                dirs = subdirs
            for idx, run_dir in enumerate(dirs):
                row = idx // 3
                col = idx % 3
                run_dict = torch.load(os.path.join(run_dir, 'run_dict'))
                weights = run_dict['model_weights']
                key = 'weight' if 'weight' in weights.keys() else '0.weight'
                current_ax = ax if n_plots == 1 else ax[idx]
                im = current_ax.imshow(weights[key], cmap='hot', interpolation='nearest', vmin=-0.8, vmax=0.8)
                if subtitle:
                    current_ax.set_title(f"run {idx+1}") # intended to be used for multiple runs within the same params
        else: # plotting weights over time for a single run/directory
            for idx, filepath in enumerate(subpaths):
                # row = idx // 3 # uncomment this soon
                # col = idx % 3
                weights = torch.load(filepath)
                key = 'weight' if 'weight' in weights.keys() else '0.weight'
                current_ax = ax if n_plots == 1 else ax[idx]
                im = current_ax.imshow(weights[key], cmap='hot', interpolation='nearest', vmin=-0.8, vmax=0.8)
                # ax[idx].title(f"run {idx+1}")
            run_dict = torch.load(os.path.join(dirs, 'run_dict'))
        if title == True:
            fig.suptitle(f"{' '.join(run_dict['essential_args'])}")
        elif title:
            fig.suptitle(title)
        # plt.subplots_adjust(hspace=0.1)
        # add colorbar
        # Adjust the spacing between subplots to make room for the colorbar
        # fig.subplots_adjust(bottom=0.15)
        cbar_ax = ax #if n_plots == 1 else ax[:, :]
        fig.colorbar(im, ax=cbar_ax, location='bottom', orientation='horizontal')
        if n_plots != 1:
            for index in range(n_plots, n_rows * n_cols):
                fig.delaxes(ax[index])
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
        weight_dict = run_dict['model_weights']
        key = 'weight' if 'weight' in weight_dict.keys() else '0.weight'
        weights = weight_dict[key]
        x1, x2 = run_dict['data']['train'][0], run_dict['data']['train'][1]
        args = run_dict["args"]

        weight_list.append(weights)
        x1_list.append(x1)
        x2_list.append(x2)

    for idx, (x1, x2, weights) in enumerate(zip(x1_list, x2_list, weight_list)):
        # calculate embedding differences
        indices = torch.randperm(x1.shape[0])[:n]
        subset1, subset2 = x1[indices], x2[indices]
        embeds1, embeds2 = subset1 @ weights.T, subset2 @ weights.T # direct embedding matrices, embeds are rows
        if idx == 0:
            print(f"Norm of {n} rows of Z1 for run {idx+1}: {torch.norm(embeds1, dim=1)}, average = {torch.mean(torch.norm(embeds1, dim=1)).item()}")
        embed_diffs = embeds1 - embeds2
        # calculate norms of differences in embeddings
        norm_diffs = torch.norm(embed_diffs, dim=1)
        embd_diff_norm_list.append(torch.mean(norm_diffs))

        # calculate (normalized) contribution of augmented dimensions to norm, if data follows nat-aug procedure
        if hasattr(args, 'k'):
            k = args.k # number of unaugmented dims, assumed all the same across all runs
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
        if hasattr(args, 'k'):
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