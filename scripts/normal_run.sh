#!/bin/bash
counter=0
num_runs=3 # number runs for each combination of hyperparams
algs=(mssla)
total_count=$((${#algs[@]} * $num_runs))

backbone=linear
optim=sgd
aug=normal
nfeats=5
featdim=5
bs=2048
lr=3e-4
sched=step
eps=100
embd=5
n=131072
save_dir="./outputs/normal_hparam_tuning3"

for i in $(seq 1 $num_runs)
do
    for alg in ${algs[@]}
    do
        ((counter++))
        echo "Run $counter of $total_count"
        echo "$alg, aug=normal, embd=5, feat_dim=5, Run $i"
        python3 pretrain.py --alg $alg --backbone $backbone --optim $optim --aug $aug --num_feats $nfeats --feat_dim $featdim --bs $bs --lr $lr --sched $sched --epochs $eps --emb_dim $embd -n $n --save_dir $save_dir
    done
done
# './outputs/normal_hparam_tuning2/normal/spectral_linear_sgd_emb_dim=5_aug=normal_lr=0.0003_sched=step_run1',


# ./outputs/normal_hparam_tuning2


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