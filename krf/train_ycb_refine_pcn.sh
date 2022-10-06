#!/bin/bash
n_gpu=6  # number of gpu to use
python3 -m torch.distributed.launch --nproc_per_node=$n_gpu train_ycb_refine_pcn.py --gpus=$n_gpu #-checkpoint '/home/zhanhz/FFB6D/ffb6d/train_log/ycb/checkpoints/FFB6D_REFINE.pth.tar'