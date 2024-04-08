#!/bin/bash

for i in {1..5}
do
    echo "Running iteration $i"
    python3 pretrain.py --alg mssl_a --backbone linear --optim adam_wd --momentum 0.9 --hidden_dim 20
done