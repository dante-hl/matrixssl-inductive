#!/bin/bash
for optim in sgd adam
do
    for lr in 1e-6 5e-6 1e-5
    do
        for wd in 1e-5 5e-6 1e-5
        do
            echo "OPTIM $optim, LR $lr, WD $wd"
            python3 pretrain.py --alg mssl_a --backbone linear --optim $optim --augmentation add --lr $lr --wd $wd --momentum 0.9 --epochs 100
        done
    done
done