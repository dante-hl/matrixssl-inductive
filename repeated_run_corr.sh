#!/bin/bash
counter=0
num_runs=5 # number runs for each combination of hyperparams
algs=(spectral)
# emb_dims=(1 2 10 11)
total_count=$((${#algs[@]} * $num_runs))

for alg in ${algs[@]}
do
    # for emb_dim in ${emb_dims[@]}
    # do
        for i in $(seq 1 $num_runs)
        do
            ((counter++))
            echo "Run $counter of $total_count"
            echo "$alg, Linear, Adam, Embedding Dimension=10, Augment=corr, y=sign(1st dim), Run $i"
            python3 pretrain.py --alg $alg --backbone linear --optim adam --emb_dim 10 --augmentation corr --lr 1e-3 --wd 1e-5 --momentum 0.9 --epochs 100
        done
    # done
done