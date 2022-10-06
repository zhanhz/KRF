#!/bin/bash
# tst_mdl=train_log/ycb/checkpoints/best/FFB6D_best.pth.tar
# python3 -m generate_ds -checkpoint $tst_mdl -dataset ycb
cls='ape'
tst_mdl="./linemod_pretrained/FFB6D_${cls}_best.pth.tar"
python3 -m generate_ds -checkpoint $tst_mdl -dataset linemod -cls ${cls} 
