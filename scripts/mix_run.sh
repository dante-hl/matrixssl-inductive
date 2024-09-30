#!/bin/bash
counter=0
num_runs=4 # number runs for each combination of hyperparams
algs=(spectral)
total_count=$((${#algs[@]} * $num_runs))

for i in $(seq 1 $num_runs)
do
    for alg in ${algs[@]}
    do
        ((counter++))
        echo "Run $counter of $total_count"
        echo "$alg, aug=mix, embd=2, Run $i"
        python3 pretrain.py --alg $alg --backbone linear --optim adam --aug mix --bs 256 --emb_dim 2 --save_dir ./outputs/mixture_gaussians
    done
done