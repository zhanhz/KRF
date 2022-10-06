#!/bin/bash
n_gpu=6
cls='holepuncher'
#ckpt_mdl="/home/zhanhz/FFB6D/ffb6d/train_log/linemod/checkpoints/${cls}/FFB6D_${cls}_REFINE_best.pth.tar"
python3 -m torch.distributed.launch --nproc_per_node=$n_gpu train_lm_refine_pcn.py --gpus=$n_gpu --cls=$cls #-checkpoint $ckpt_mdl
