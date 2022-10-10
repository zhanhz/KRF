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

config = Config(ds_name='ycb')
bs_utils = Basic_Utils(config)
cls_lst = config.ycb_cls_lst
class TorchEval():

    def __init__(self):
        n_cls = 22
        self.n_cls = 22
        self.cls_add_dis = [list() for i in range(n_cls)]
        self.cls_adds_dis = [list() for i in range(n_cls)]
        self.cls_add_s_dis = [list() for i in range(n_cls)]
        self.pred_kp_errs = [list() for i in range(n_cls)]

        self.cls_add_dis_icp = [list() for i in range(n_cls)]
        self.cls_adds_dis_icp = [list() for i in range(n_cls)]
        self.cls_add_s_dis_icp = [list() for i in range(n_cls)]
        self.pred_id2pose_lst = []
        self.sym_cls_ids = []

    def cal_auc(self):
        add_auc_lst = []
        adds_auc_lst = []
        add_s_auc_lst = []

        add_auc_lst_icp = []
        adds_auc_lst_icp = []
        add_s_auc_lst_icp = []
        for cls_id in range(1, self.n_cls):
            if (cls_id) in config.ycb_sym_cls_ids:
                self.cls_add_s_dis[cls_id] = self.cls_adds_dis[cls_id]
                self.cls_add_s_dis_icp[cls_id] = self.cls_adds_dis_icp[cls_id]
            else:
                self.cls_add_s_dis[cls_id] = self.cls_add_dis[cls_id]
                self.cls_add_s_dis_icp[cls_id] = self.cls_add_dis_icp[cls_id]
            self.cls_add_s_dis[0] += self.cls_add_s_dis[cls_id]
            self.cls_add_s_dis_icp[0] += self.cls_add_s_dis_icp[cls_id]
        for i in range(self.n_cls):
            add_auc = bs_utils.cal_auc(self.cls_add_dis[i])
            adds_auc = bs_utils.cal_auc(self.cls_adds_dis[i])
            add_s_auc = bs_utils.cal_auc(self.cls_add_s_dis[i])
            add_auc_lst.append(add_auc)
            adds_auc_lst.append(adds_auc)
            add_s_auc_lst.append(add_s_auc)


            add_auc_icp = bs_utils.cal_auc(self.cls_add_dis_icp[i])
            adds_auc_icp = bs_utils.cal_auc(self.cls_adds_dis_icp[i])
            add_s_auc_icp = bs_utils.cal_auc(self.cls_add_s_dis_icp[i])
            add_auc_lst_icp.append(add_auc_icp)
            adds_auc_lst_icp.append(adds_auc_icp)
            add_s_auc_lst_icp.append(add_s_auc_icp)
            if i == 0:
                continue
            print(cls_lst[i-1])
            print("***************add:\t", add_auc)
            print("***************adds:\t", adds_auc)
            print("***************add(-s):\t", add_s_auc)
            print("***************add_icp:\t", add_auc_icp)
            print("***************adds_icp:\t", adds_auc_icp)
            print("***************add(-s)_icp:\t", add_s_auc_icp)
        # kp errs:
        n_objs = sum([len(l) for l in self.pred_kp_errs])
        all_errs = 0.0
        for cls_id in range(1, self.n_cls):
            all_errs += sum(self.pred_kp_errs[cls_id])
        print("mean kps errs:", all_errs / n_objs)

        print("Average of all object:")
        print("BaseLine:")
        print("***************add:\t", np.mean(add_auc_lst[1:]))
        print("***************adds:\t", np.mean(adds_auc_lst[1:]))
        print("***************add(-s):\t", np.mean(add_s_auc_lst[1:]))

        print("Ours:")
        print("***************add:\t", np.mean(add_auc_lst_icp[1:]))
        print("***************adds:\t", np.mean(adds_auc_lst_icp[1:]))
        print("***************add(-s):\t", np.mean(add_s_auc_lst_icp[1:]))

        print("All object (following PoseCNN):")
        print("BaseLine:")
        print("***************add:\t", add_auc_lst[0])
        print("***************adds:\t", adds_auc_lst[0])
        print("***************add(-s):\t", add_s_auc_lst[0])

        print("Ours:")
        print("***************add:\t", add_auc_lst_icp[0])
        print("***************adds:\t", adds_auc_lst_icp[0])
        print("***************add(-s):\t", add_s_auc_lst_icp[0])

        sv_info = dict(
            add_dis_lst=self.cls_add_dis,
            adds_dis_lst=self.cls_adds_dis,
            add_auc_lst=add_auc_lst,
            adds_auc_lst=adds_auc_lst,
            add_s_auc_lst=add_s_auc_lst,
            pred_kp_errs=self.pred_kp_errs,
        )
        sv_pth = os.path.join(
            config.log_eval_dir,
            'pvn3d_eval_cuda_{}_{}_{}.pkl'.format(
                adds_auc_lst[0], add_auc_lst[0], add_s_auc_lst[0]
            )
        )
        pkl.dump(sv_info, open(sv_pth, 'wb'))
        sv_pth = os.path.join(
            config.log_eval_dir,
            'pvn3d_eval_cuda_{}_{}_{}_id2pose.pkl'.format(
                adds_auc_lst[0], add_auc_lst[0], add_s_auc_lst[0]
            )
        )
        pkl.dump(self.pred_id2pose_lst, open(sv_pth, 'wb'))


    def push(self, cls_add_dis_lst, cls_adds_dis_lst, cls_add_dis_icp_lst, cls_adds_dis_icp_lst, pred_cls_ids, pred_poses, pred_kp_errs):
        self.pred_id2pose_lst.append(
            {cid: pose for cid, pose in zip(pred_cls_ids, pred_poses)}
        )
        self.pred_kp_errs = self.merge_lst(
            self.pred_kp_errs, pred_kp_errs
        )
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