#!/bin/bash
counter=0
num_runs=2 # number runs for each combination of hyperparams
algs=(spectral)
total_count=$((${#algs[@]} * $num_runs))

for i in $(seq 1 $num_runs)
do
    for alg in ${algs[@]}
    do
        ((counter++))
        echo "Run $counter of $total_count"
        echo "$alg, aug=normal, embd=5, feat_dim=5, Run $i"
        python3 pretrain.py --alg $alg --backbone linear --optim sgd --aug normal --num_feats 5 --feat_dim 5 --bs 256 --lr 3e-4 --sched step --epochs 100 --emb_dim 5 --save_dir ./outputs/normal_hparam_tuning2
    done
done
# defaults in pretrain.py:
# n (2 ** 16)
# num_feats 5
# feat_dim 5
# embd 10
# epochs 100
# bs 256
# lr 1e-3
# wd 1e-5
# momentum 0.9