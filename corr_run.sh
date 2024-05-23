#!/bin/bash
counter=0
num_runs=2 # number runs for each combination of hyperparams
algs=(spectral mssla)
total_count=$((${#algs[@]} * $num_runs))

for i in $(seq 1 $num_runs)
do
    for alg in ${algs[@]}
    do
        ((counter++))
        echo "Run $counter of $total_count"
        echo "$alg, nat=bool, aug=corr, label=spur, embd=15, tau_max=1.5, Run $i"
        python3 pretrain.py --alg $alg --backbone linear --optim adam --nat bool --aug corr --label spur --bs 256 -d 25 -k 5 --emb_dim 15 --save_dir ./outputs/embd15_runs
    done
done
# defaults in pretrain.py:
# consider removing these args from CLI run above, if already set to default
# n (2 ** 16) + 12500
# v 12500
# d 50
# k 10
# embd 20
# epochs 100
# bs 128
# lr 1e-3
# wd 1e-5
# momentum 0.9