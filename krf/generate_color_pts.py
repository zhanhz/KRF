import os
import numpy as np
import cv2
import torch
from sklearn.neighbors import NearestNeighbors

def read_obj(pth):
    f = open(pth)
    xyz=[]
    color=[]
    while(True):
        line = f.readline().strip()
        if not line:
            break
        elif line[0] == 'v':
            line = line.split(' ')
            xyz.append(np.float32(line[1:4]))
            color.append(np.float32(line[4:]))
        else:
            continue
    return np.array(xyz), np.array(color)

root_path ='./datasets/ycb/YCB_Video_Dataset/models'
obj_cls = os.listdir(root_path)
for c in obj_cls:
    pts_path = os.path.join(root_path, c, 'points.xyz')
    xyzrgb_path = os.path.join(root_path, c, 'xyzrgb.obj')
    pts = np.loadtxt(pts_path, dtype=np.float32)
    xyz, color = read_obj(xyzrgb_path)

    choose = np.zeros(xyz.shape[0])
    choose[:8000] = 1
    np.random.shuffle(choose)
    xyz = xyz[np.nonzero(choose)]
    color = color[np.nonzero(choose)]
    pts_clr = np.concatenate([xyz, color], axis=-1)
    np.savetxt(os.path.join(root_path, c, 'pts_clr_8000.txt'), pts_clr, fmt='%.6f')
    print(c + ' complete')
