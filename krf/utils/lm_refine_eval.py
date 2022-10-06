#!/usr/bin/env python3
import os
import torch
import os.path
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from common import Config
import pickle as pkl
from utils.basic_utils import Basic_Utils
import scipy.io as scio
import scipy.misc

config = Config(ds_name='linemod')
bs_utils = Basic_Utils(config)
class TorchEval():

    def __init__(self):
        n_cls = 2
        self.n_cls = 2
        self.cls_add_dis = [list() for i in range(n_cls)]
        self.cls_adds_dis = [list() for i in range(n_cls)]
        self.cls_add_s_dis = [list() for i in range(n_cls)]
        self.pred_kp_errs = [list() for i in range(n_cls)]

        self.cls_add_dis_icp = [list() for i in range(n_cls)]
        self.cls_adds_dis_icp = [list() for i in range(n_cls)]
        self.cls_add_s_dis_icp = [list() for i in range(n_cls)]
        self.pred_id2pose_lst = []
        self.sym_cls_ids = []

    def cal_auc(self, obj_id):
        add_auc_lst = []
        adds_auc_lst = []
        add_s_auc_lst = []

        add_auc_lst_icp = []
        adds_auc_lst_icp = []
        add_s_auc_lst_icp = []
        cls_id = 1
        
        if (obj_id) in config.lm_sym_cls_ids:
            self.cls_add_s_dis[cls_id] = self.cls_adds_dis[cls_id]
            self.cls_add_s_dis_icp[cls_id] = self.cls_adds_dis_icp[cls_id]
        else:
            self.cls_add_s_dis[cls_id] = self.cls_add_dis[cls_id]
            self.cls_add_s_dis_icp[cls_id] = self.cls_add_dis_icp[cls_id]

        self.cls_add_s_dis[0] += self.cls_add_s_dis[cls_id]
        self.cls_add_s_dis_icp[0] += self.cls_add_s_dis_icp[cls_id]

        add_auc = bs_utils.cal_auc(self.cls_add_dis[cls_id])
        adds_auc = bs_utils.cal_auc(self.cls_adds_dis[cls_id])
        add_s_auc = bs_utils.cal_auc(self.cls_add_s_dis[cls_id])
        add_auc_lst.append(add_auc)
        adds_auc_lst.append(adds_auc)
        add_s_auc_lst.append(add_s_auc)


        add_auc_icp = bs_utils.cal_auc(self.cls_add_dis_icp[cls_id])
        adds_auc_icp = bs_utils.cal_auc(self.cls_adds_dis_icp[cls_id])
        add_s_auc_icp = bs_utils.cal_auc(self.cls_add_s_dis_icp[cls_id])
        add_auc_lst_icp.append(add_auc_icp)
        adds_auc_lst_icp.append(adds_auc_icp)
        add_s_auc_lst_icp.append(add_s_auc_icp)
        
        d = config.lm_r_lst[obj_id]['diameter'] / 1000.0 * 0.1
        add = np.mean(np.array(self.cls_add_dis[cls_id]) < d) * 100
        adds = np.mean(np.array(self.cls_adds_dis[cls_id]) < d) * 100
        add_icp = np.mean(np.array(self.cls_add_dis_icp[cls_id]) < d) * 100
        adds_icp = np.mean(np.array(self.cls_adds_dis_icp[cls_id]) < d) * 100
        

        print("obj_id: ", obj_id, "0.1 diameter: ", d)

        cls_type = config.lm_id2obj_dict[obj_id]
        print(obj_id, cls_type)
        print("***************add:\t", add_auc)
        print("***************adds:\t", adds_auc)
        print("***************add(-s):\t", add_s_auc)
        print("***************add < 0.1 diameter:\t", add)
        print("***************adds < 0.1 diameter:\t", adds)

        print("***************add_icp:\t", add_auc_icp)
        print("***************adds_icp:\t", adds_auc_icp)
        print("***************add(-s)_icp:\t", add_s_auc_icp)
        print("***************add_icp < 0.1 diameter:\t", add_icp)
        print("***************adds_icp < 0.1 diameter:\t", adds_icp)


    def push(self, cls_add_dis_lst, cls_adds_dis_lst, cls_add_dis_icp_lst, cls_adds_dis_icp_lst, pred_poses):
        self.cls_add_dis = self.merge_lst(
            self.cls_add_dis, cls_add_dis_lst
        )
        self.cls_adds_dis = self.merge_lst(
            self.cls_adds_dis, cls_adds_dis_lst
        )
        self.cls_add_dis_icp = self.merge_lst(
            self.cls_add_dis_icp, cls_add_dis_icp_lst
        )
        self.cls_adds_dis_icp = self.merge_lst(
            self.cls_adds_dis_icp, cls_adds_dis_icp_lst
        )                
    def merge_lst(self, targ, src):
        for i in range(len(targ)):
            targ[i] += src[i]
        return targ