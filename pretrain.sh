#!/bin/bash

python main_pretrain.py \
  -a resnet50 \
  --dataset CIFAR10 \
  --dist-url 'tcp://localhost:10501' --multiprocessing-distributed --world-size 1 --rank 0 \
  --workers 4 \
  --model mec --loss_type mce \
  --wandb_logging all --project_name mssl-inductive --run run1 \
  --save_dir "./outputs/mec_pretrain" \
  /workspace/projects/matrixssl-inductive/datasets/cifar10