#!/usr/bin/env python3
import os
import cv2
import torch
import os.path
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from common import Config
import pickle as pkl
from utils.basic_utils import Basic_Utils
import yaml
import scipy.io as scio
import scipy.misc
from glob import glob
from termcolor import colored
import normalSpeed
from models.RandLA.helper_tool import DataProcessing as DP
try:
    from neupeak.utils.webcv2 import imshow, waitKey
except ImportError:
    from cv2 import imshow, waitKey
from knn_cuda import KNN
def knn_cuda(ref, query, k):
    ref = torch.from_numpy(ref).cuda()
    query = torch.from_numpy(query).cuda()
    m = ref.shape[-1]
    knn = KNN(k = k, transpose_mode=True)
    # ref = ref.reshape((1, -1, m))
    # query = query.reshape((1, -1, m))
    dist, idx = knn(ref, query)
    return idx.cpu().numpy()
class Dataset():

    def __init__(self, dataset_name, cls_type="duck", DEBUG=False):
        self.DEBUG = DEBUG
        self.config = Config(ds_name='linemod', cls_type=cls_type)
        self.bs_utils = Basic_Utils(self.config)
        self.dataset_name = dataset_name
        self.xmap = np.array([[j for i in range(640)] for j in range(480)])
        self.ymap = np.array([[i for i in range(640)] for j in range(480)])

        self.trancolor = transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)
        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.224])
        self.obj_dict = self.config.lm_obj_dict

        self.cls_type = cls_type
        self.cls_id = self.obj_dict[cls_type]
        print("cls_id in lm_dataset.py", self.cls_id)
        self.root = os.path.join(self.config.lm_root, 'Linemod_preprocessed')
        # self.cls_root = os.path.join(self.root, "data/%02d/" % self.cls_id)
        self.cls_root = os.path.join(self.root, "data/02/")
        self.rng = np.random
        meta_file = open(os.path.join(self.cls_root, 'gt.yml'), "r")
        self.meta_lst = yaml.load(meta_file)
        if dataset_name == 'train':
            self.add_noise = True
            real_img_pth = os.path.join(
                self.cls_root, "train.txt"
            )
            self.real_lst = self.bs_utils.read_lines(real_img_pth)

            rnd_img_ptn = os.path.join(
                self.root, 'renders/%s/*.pkl' % cls_type
            )
            self.rnd_lst = glob(rnd_img_ptn)
            print("render data length: ", len(self.rnd_lst))
            if len(self.rnd_lst) == 0:
                warning = "Warning: "
                warning += "Trainnig without rendered data will hurt model performance \n"
                warning += "Please generate rendered data from https://github.com/ethnhe/raster_triangle.\n"
                print(colored(warning, "red", attrs=['bold']))

            fuse_img_ptn = os.path.join(
                self.root, 'fuse/%s/*.pkl' % cls_type
            )
            self.fuse_lst = glob(fuse_img_ptn)
            print("fused data length: ", len(self.fuse_lst))
            if len(self.fuse_lst) == 0:
                warning = "Warning: "
                warning += "Trainnig without fused data will hurt model performance \n"
                warning += "Please generate fused data from https://github.com/ethnhe/raster_triangle.\n"
                print(colored(warning, "red", attrs=['bold']))

            self.all_lst = self.real_lst + self.rnd_lst + self.fuse_lst
            self.minibatch_per_epoch = len(self.all_lst) // self.config.mini_batch_size
        else:
            self.add_noise = False
            tst_img_pth = os.path.join(self.cls_root, "test_occ.txt")
            # tst_img_pth = os.path.join(self.cls_root, "test.txt")
            self.tst_lst = self.bs_utils.read_lines(tst_img_pth)
            self.all_lst = self.tst_lst
        print("{}_dataset_size: ".format(dataset_name), len(self.all_lst))

    def real_syn_gen(self, real_ratio=0.3):
        if len(self.rnd_lst+self.fuse_lst) == 0:
            real_ratio = 1.0
        if self.rng.rand() < real_ratio:  # real
            n_imgs = len(self.real_lst)
            idx = self.rng.randint(0, n_imgs)
            pth = self.real_lst[idx]
            return pth
        else:
            if len(self.fuse_lst) > 0 and len(self.rnd_lst) > 0:
                fuse_ratio = 0.4
            elif len(self.fuse_lst) == 0:
                fuse_ratio = 0.
            else:
                fuse_ratio = 1.
            if self.rng.rand() < fuse_ratio:
                idx = self.rng.randint(0, len(self.fuse_lst))
                pth = self.fuse_lst[idx]
            else:
                idx = self.rng.randint(0, len(self.rnd_lst))
                pth = self.rnd_lst[idx]
            return pth

    def real_gen(self):
        n = len(self.real_lst)
        idx = self.rng.randint(0, n)
        item = self.real_lst[idx]
        return item

    def rand_range(self, rng, lo, hi):
        return rng.rand()*(hi-lo)+lo

    def gaussian_noise(self, rng, img, sigma):
        """add gaussian noise of given sigma to image"""
        img = img + rng.randn(*img.shape) * sigma
        img = np.clip(img, 0, 255).astype('uint8')
        return img

    def linear_motion_blur(self, img, angle, length):
        """:param angle: in degree"""
        rad = np.deg2rad(angle)
        dx = np.cos(rad)
        dy = np.sin(rad)
        a = int(max(list(map(abs, (dx, dy)))) * length * 2)
        if a <= 0:
            return img
        kern = np.zeros((a, a))
        cx, cy = a // 2, a // 2
        dx, dy = list(map(int, (dx * length + cx, dy * length + cy)))
        cv2.line(kern, (cx, cy), (dx, dy), 1.0)
        s = kern.sum()
        if s == 0:
            kern[cx, cy] = 1.0
        else:
            kern /= s
        return cv2.filter2D(img, -1, kern)

    def rgb_add_noise(self, img):
        rng = self.rng
        # apply HSV augmentor
        if rng.rand() > 0:
            hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.uint16)
            hsv_img[:, :, 1] = hsv_img[:, :, 1] * self.rand_range(rng, 1-0.25, 1+.25)
            hsv_img[:, :, 2] = hsv_img[:, :, 2] * self.rand_range(rng, 1-.15, 1+.15)
            hsv_img[:, :, 1] = np.clip(hsv_img[:, :, 1], 0, 255)
            hsv_img[:, :, 2] = np.clip(hsv_img[:, :, 2], 0, 255)
            img = cv2.cvtColor(hsv_img.astype(np.uint8), cv2.COLOR_HSV2BGR)

        if rng.rand() > 0.8:  # motion blur
            r_angle = int(rng.rand() * 360)
            r_len = int(rng.rand() * 15) + 1
            img = self.linear_motion_blur(img, r_angle, r_len)

        if rng.rand() > 0.8:
            if rng.rand() > 0.2:
                img = cv2.GaussianBlur(img, (3, 3), rng.rand())
            else:
                img = cv2.GaussianBlur(img, (5, 5), rng.rand())

        return np.clip(img, 0, 255).astype(np.uint8)

    def add_real_back(self, rgb, labels, dpt, dpt_msk):
        real_item = self.real_gen()
        with Image.open(os.path.join(self.cls_root, "depth", real_item+'.png')) as di:
            real_dpt = np.array(di)
        with Image.open(os.path.join(self.cls_root, "mask", real_item+'.png')) as li:
            bk_label = np.array(li)
        bk_label = (bk_label < 255).astype(rgb.dtype)
        if len(bk_label.shape) > 2:
            bk_label = bk_label[:, :, 0]
        with Image.open(os.path.join(self.cls_root, "rgb", real_item+'.png')) as ri:
            back = np.array(ri)[:, :, :3] * bk_label[:, :, None]
        dpt_back = real_dpt.astype(np.float32) * bk_label.astype(np.float32)

        if self.rng.rand() < 0.6:
            msk_back = (labels <= 0).astype(rgb.dtype)
            msk_back = msk_back[:, :, None]
            rgb = rgb * (msk_back == 0).astype(rgb.dtype) + back * msk_back

        dpt = dpt * (dpt_msk > 0).astype(dpt.dtype) + \
            dpt_back * (dpt_msk <= 0).astype(dpt.dtype)
        return rgb, dpt

    def dpt_2_pcld(self, dpt, rgb, cam_scale, K):
        if len(dpt.shape) > 2:
            dpt = dpt[:, :, 0]
        dpt = dpt.astype(np.float32) / cam_scale
        msk = (dpt > 1e-8).astype(np.float32)
        choose = (dpt > 1e-8).flatten().nonzero()[0].astype(np.uint32)
        row = (self.ymap - K[0][2]) * dpt / K[0][0]
        col = (self.xmap - K[1][2]) * dpt / K[1][1]
        dpt_3d = np.concatenate(
            (row[..., None], col[..., None], dpt[..., None]), axis=2
        )
        dpt_3d = dpt_3d * msk[:, :, None]
        rgb_pts = rgb * msk[:, :, None].astype(np.float32)
        return dpt_3d, rgb_pts, choose

    def get_item(self, item_name, pred_pose):
        if "pkl" in item_name:
            data = pkl.load(open(item_name, "rb"))
            dpt_mm = data['depth'] * 1000.
            rgb = data['rgb']
            labels = data['mask']
            K = data['K']
            RT = data['RT']
            rnd_typ = data['rnd_typ']
            name = item_name.split('/')[-1][:-4]
            if rnd_typ == "fuse":
                labels = (labels == self.cls_id).astype("uint8")
            else:
                labels = (labels > 0).astype("uint8")
            with Image.open(os.path.join(self.cls_root, "add_data/{}/pred_label/{}.png".format(rnd_typ, name))) as pli:
                pred_labels = np.array(pli)
                pred_labels = (pred_labels > 0).astype("uint8")
        else:
            occ_root ='/home/zhanhz/FFB6D/ffb6d/datasets/linemod/Linemod_preprocessed/occ_data/%02d' % self.cls_id
            name = item_name
            with Image.open(os.path.join(self.cls_root, "depth/{}.png".format(item_name))) as di:
                dpt_mm = np.array(di)

            with Image.open(os.path.join(self.cls_root, "mask_all/{}.png".format(item_name))) as li:
                labels = np.array(li)
                labels = (labels == round(255.0 / float(self.cls_id))).astype("uint8")

            with Image.open(os.path.join(occ_root, "add_data/pred_label/{}.png".format(item_name))) as pli:
                pred_labels = np.array(pli)
                pred_labels = (pred_labels > 0).astype("uint8")
                if (np.nonzero(pred_labels)[0].shape[0] == 0): return None
            with Image.open(os.path.join(self.cls_root, "rgb/{}.png".format(item_name))) as ri:
                if self.add_noise:
                    ri = self.trancolor(ri)
                rgb = np.array(ri)[:, :, :3]
            meta = self.meta_lst[int(item_name)]
            if self.cls_id == 2:
                for i in range(0, len(meta)):
                    if meta[i]['obj_id'] == 2:
                        meta = meta[i]
                        break
            else:
                meta = meta[0]
            R = np.resize(np.array(meta['cam_R_m2c']), (3, 3))
            T = np.array(meta['cam_t_m2c']) / 1000.0
            RT = np.concatenate((R, T[:, None]), axis=1)
            rnd_typ = 'real'
            K = self.config.intrinsic_matrix["linemod"]
        cam_scale = 1000.0
        if len(labels.shape) > 2:
            labels = labels[:, :, 0]
        rgb_labels = labels.copy()
        pred_rgb_labels = pred_labels.copy()

        if self.add_noise and rnd_typ != 'real':
            if rnd_typ == 'render' or self.rng.rand() < 0.8:
                rgb = self.rgb_add_noise(rgb)
                rgb_labels = labels.copy()
                msk_dp = dpt_mm > 1e-6
                rgb, dpt_mm = self.add_real_back(rgb, rgb_labels, dpt_mm, msk_dp)
                if self.rng.rand() > 0.8:
                    rgb = self.rgb_add_noise(rgb)

        dpt_mm = dpt_mm.copy().astype(np.uint16)
        nrm_map = normalSpeed.depth_normal(
            dpt_mm, K[0][0], K[1][1], 5, 2000, 20, False
        )
        if self.DEBUG:
            show_nrm_map = ((nrm_map + 1.0) * 127).astype(np.uint8)
            imshow("nrm_map", show_nrm_map)

        dpt_m = dpt_mm.astype(np.float32) / cam_scale
        dpt_xyz_ori, rgb_pts_all, all_choose = self.dpt_2_pcld(dpt_m, rgb, 1.0, K)
        dpt_xyz_ori[np.isnan(dpt_xyz_ori)] = 0.0
        dpt_xyz_ori[np.isinf(dpt_xyz_ori)] = 0.0

        all_cld = dpt_xyz_ori.reshape(-1,3)
        rgb_pts_all = rgb_pts_all.reshape(-1,3)

        choose = np.arange(0, len(all_cld))
        msk_obj = (pred_rgb_labels.flatten() > 0)
        if self.config.n_keypoints == 8:
            kp_type = 'farthest'
        else:
            kp_type = 'farthest{}'.format(self.config.n_keypoints)
        kps = self.bs_utils.get_kps(
            self.cls_type, kp_type=kp_type, ds_type='linemod'
        )
        r = pred_pose[:, :3]
        t = pred_pose[:, 3]
        r1 = np.dot(R.T, r)
        t1 = np.dot(T[:, None].T - t, r)
        kps_gt = np.dot(kps, r1) + t1
        dpt_xyz = np.dot(all_cld - t, r).reshape(dpt_xyz_ori.shape)
        obj_cld = dpt_xyz.reshape(-1,3)[msk_obj]
        all_cld = all_cld[msk_obj]
        rgb_pts_all = rgb_pts_all[msk_obj]

        n_sample = 2048
        if obj_cld.shape[0] > n_sample // 4:
            kpcld_idx = np.random.permutation(obj_cld.shape[0])
            if kpcld_idx.shape[0] < n_sample:
                kpcld_idx = np.concatenate([kpcld_idx, np.random.randint(kpcld_idx.shape[0], size=n_sample - kpcld_idx.shape[0])])
            kpcld_idx = kpcld_idx[:n_sample]
            cld_kp = obj_cld[kpcld_idx]
            kps_gt_rp = np.repeat(kps_gt[np.newaxis,:], cld_kp.shape[0], axis=0)
            cld_kp_rp = np.repeat(cld_kp[:, np.newaxis, :], self.config.n_keypoints, axis=1)
            kp_targ_ofst = kps_gt_rp - cld_kp_rp
            choose = choose[msk_obj][kpcld_idx]
        else: return None

        cld = cld_kp
        rgb_pt = rgb.reshape(-1, 3)[choose, :].astype(np.float32)
        nrm_pt = nrm_map[:, :, :3].reshape(-1, 3)[choose, :]
        labels_pt = labels.flatten()[choose]
        pred_labels_pt = pred_labels.flatten()[choose]
        choose = np.array([choose])
        cld_rgb_nrm = np.concatenate((cld, rgb_pt, nrm_pt), axis=1).transpose(1, 0)


        h, w = rgb_labels.shape
        rgb = np.transpose(rgb, (2, 0, 1)) # hwc2chw
        xyz_lst = [dpt_xyz.transpose(2, 0, 1)] # c, h, w

        for i in range(3):
            scale = pow(2, i+1)
            nh, nw = h // pow(2, i+1), w // pow(2, i+1)
            ys, xs = np.mgrid[:nh, :nw]
            xyz_lst.append(xyz_lst[0][:, ys*scale, xs*scale])
        sr2dptxyz = {
            pow(2, ii): item.reshape(3, -1).transpose(1, 0)
            for ii, item in enumerate(xyz_lst)
        }

        rgb_ds_sr = [4, 8, 8, 8]
        n_ds_layers = 4
        pcld_sub_s_r = [4, 4, 4, 4]
        inputs = {}
        # DownSample stage
        for i in range(n_ds_layers):
            nei_idx = knn_cuda(
                cld[None, ...], cld[None, ...], 16
            ).astype(np.int32).squeeze(0)
            sub_pts = cld[:cld.shape[0] // pcld_sub_s_r[i], :]
            pool_i = nei_idx[:cld.shape[0] // pcld_sub_s_r[i], :]
            up_i = knn_cuda(
                sub_pts[None, ...], cld[None, ...], 1
            ).astype(np.int32).squeeze(0)
            inputs['cld_xyz%d'%i] = cld.astype(np.float32).copy()
            inputs['cld_nei_idx%d'%i] = nei_idx.astype(np.int32).copy()
            inputs['cld_sub_idx%d'%i] = pool_i.astype(np.int32).copy()
            inputs['cld_interp_idx%d'%i] = up_i.astype(np.int32).copy()
            nei_r2p = knn_cuda(
                sr2dptxyz[rgb_ds_sr[i]][None, ...], sub_pts[None, ...], 16
            ).astype(np.int32).squeeze(0)
            inputs['r2p_ds_nei_idx%d'%i] = nei_r2p.copy()
            nei_p2r = knn_cuda(
                sub_pts[None, ...], sr2dptxyz[rgb_ds_sr[i]][None, ...], 1
            ).astype(np.int32).squeeze(0)
            inputs['p2r_ds_nei_idx%d'%i] = nei_p2r.copy()
            cld = sub_pts

        n_up_layers = 3
        rgb_up_sr = [4, 2, 2]
        for i in range(n_up_layers):
            r2p_nei = knn_cuda(
                sr2dptxyz[rgb_up_sr[i]][None, ...],
                inputs['cld_xyz%d'%(n_ds_layers-i-1)][None, ...], 16
            ).astype(np.int32).squeeze(0)
            inputs['r2p_up_nei_idx%d'%i] = r2p_nei.copy()
            p2r_nei = knn_cuda(
                inputs['cld_xyz%d'%(n_ds_layers-i-1)][None, ...],
                sr2dptxyz[rgb_up_sr[i]][None, ...], 1
            ).astype(np.int32).squeeze(0)
            inputs['p2r_up_nei_idx%d'%i] = p2r_nei.copy()

        item_dict = dict(
            rgb=rgb.astype(np.uint8),  # [c, h, w]
            cld_rgb_nrm=cld_rgb_nrm.astype(np.float32),  # [9, npts]
            choose=choose.astype(np.int32),  # [1, npts]
            labels=labels_pt.astype(np.int32),  # [npts]
            pred_labels=pred_labels_pt.astype(np.int32),  # [npts]
            rgb_labels=rgb_labels.astype(np.int32),  # [h, w]
            all_cld=all_cld.astype(np.float32), #added by ourself
            kp_targ_ofst=kp_targ_ofst.astype(np.float32),
            rgb_pt=rgb_pts_all.reshape(-1,3).astype(np.float32), #added by ourself
        )
        item_dict.update(inputs)
        for key in item_dict.keys():
            item_dict[key] = item_dict[key][None, ...]
        return item_dict
# vim: ts=4 sw=4 sts=4 expandtab
