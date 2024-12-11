python main_lincls.py \
  -a resnet50 \
  --dist-url 'tcp://localhost:10051' --multiprocessing-distributed --world-size 1 --rank 0 \
  --pretrained '/projects/matrixssl-inductive/outputs/mec_pretrain/checkpoint_0099.pth.tar' \
  --lars \
  /workspace/projects/matrixssl-inductive/datasets/cifar10
