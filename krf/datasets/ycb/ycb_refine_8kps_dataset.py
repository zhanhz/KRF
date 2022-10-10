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
import scipy.io as scio
import scipy.misc
try:
    from neupeak.utils.webcv2 import imshow, waitKey
except:
    from cv2 import imshow, waitKey
import normalSpeed
from models.RandLA.helper_tool import DataProcessing as DP


config = Config(ds_name='ycb')
bs_utils = Basic_Utils(config)


class Dataset():

    def __init__(self, dataset_name, DEBUG=False):
        self.dataset_name = dataset_name
        self.debug = DEBUG
        self.xmap = np.array([[j for i in range(640)] for j in range(480)])
        self.ymap = np.array([[i for i in range(640)] for j in range(480)])
        self.diameters = {}
        self.trancolor = transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)
        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.224])
        self.cls_lst = bs_utils.read_lines(config.ycb_cls_lst_p)
        self.obj_dict = {}
        for cls_id, cls in enumerate(self.cls_lst, start=1):
            self.obj_dict[cls] = cls_id
        self.rng = np.random
        if dataset_name == 'train':
            self.add_noise = True
            self.path = 'datasets/ycb/dataset_config/train_data_list.txt'
            self.all_lst = bs_utils.read_lines(self.path)
            self.len = len(self.all_lst)
            self.minibatch_per_epoch = self.len // config.mini_batch_size
            self.real_lst = []
            self.syn_lst = []
            for item in self.all_lst:
                if item[:5] == 'data/':
                    self.real_lst.append(item)
                else:
                    self.syn_lst.append(item)
        else:
            self.pp_data = None
            self.add_noise = False
            self.path = 'datasets/ycb/dataset_config/test_data_list.txt'
            self.all_lst = bs_utils.read_lines(self.path)
            self.len = len(self.all_lst)
        print("{}_dataset_size: ".format(dataset_name), self.len)
        self.root = config.ycb_root
        self.pred_root = os.path.join(self.root, 'add_data')
        self.sym_cls_ids = [13, 16, 19, 20, 21]

    def real_syn_gen(self):
        if self.rng.rand() > 0.8:
            n = len(self.real_lst)
            idx = self.rng.randint(0, n)
            item = self.real_lst[idx]
        else:
            n = len(self.syn_lst)
            idx = self.rng.randint(0, n)
            item = self.syn_lst[idx]
        return item

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
            hsv_img[:, :, 1] = hsv_img[:, :, 1] * self.rand_range(rng, 1.25, 1.45)
            hsv_img[:, :, 2] = hsv_img[:, :, 2] * self.rand_range(rng, 1.15, 1.35)
            hsv_img[:, :, 1] = np.clip(hsv_img[:, :, 1], 0, 255)
            hsv_img[:, :, 2] = np.clip(hsv_img[:, :, 2], 0, 255)
            img = cv2.cvtColor(hsv_img.astype(np.uint8), cv2.COLOR_HSV2BGR)

        if rng.rand() > .8:  # sharpen
            kernel = -np.ones((3, 3))
            kernel[1, 1] = rng.rand() * 3 + 9
            kernel /= kernel.sum()
            img = cv2.filter2D(img, -1, kernel)

        if rng.rand() > 0.8:  # motion blur
            r_angle = int(rng.rand() * 360)
            r_len = int(rng.rand() * 15) + 1
            img = self.linear_motion_blur(img, r_angle, r_len)

        if rng.rand() > 0.8:
            if rng.rand() > 0.2:
                img = cv2.GaussianBlur(img, (3, 3), rng.rand())
            else:
                img = cv2.GaussianBlur(img, (5, 5), rng.rand())

        if rng.rand() > 0.2:
            img = self.gaussian_noise(rng, img, rng.randint(15))
        else:
            img = self.gaussian_noise(rng, img, rng.randint(25))

        if rng.rand() > 0.8:
            img = img + np.random.normal(loc=0.0, scale=7.0, size=img.shape)

        return np.clip(img, 0, 255).astype(np.uint8)

    def add_real_back(self, rgb, labels, dpt, dpt_msk):
        real_item = self.real_gen()
        with Image.open(os.path.join(self.root, real_item+'-depth.png')) as di:
            real_dpt = np.array(di)
        with Image.open(os.path.join(self.root, real_item+'-label.png')) as li:
            bk_label = np.array(li)
        bk_label = (bk_label <= 0).astype(rgb.dtype)
        bk_label_3c = np.repeat(bk_label[:, :, None], 3, 2)
        with Image.open(os.path.join(self.root, real_item+'-color.png')) as ri:
            back = np.array(ri)[:, :, :3] * bk_label_3c
        dpt_back = real_dpt.astype(np.float32) * bk_label.astype(np.float32)

        msk_back = (labels <= 0).astype(rgb.dtype)
        msk_back = np.repeat(msk_back[:, :, None], 3, 2)
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

    def get_item(self, item_name):
        with Image.open(os.path.join(self.root, item_name+'-depth.png')) as di:
            dpt_um = np.array(di)
        with Image.open(os.path.join(self.root, item_name+'-label.png')) as li:
            labels = np.array(li)
        with Image.open(os.path.join(self.pred_root, item_name+ '-pred_label.png')) as pli:
            pred_labels = np.array(pli)
        rgb_labels = labels.copy()
        pred_rgb_labels = pred_labels.copy()
        pred_pose = np.load(os.path.join(self.pred_root, item_name+'-pred_pose.npy'))
        meta = scio.loadmat(os.path.join(self.root, item_name+'-meta.mat'))
        cls_id_lst = meta['cls_indexes'].flatten().astype(np.uint32)



        if item_name[:8] != 'data_syn' and int(item_name[5:9]) >= 60:
            K = config.intrinsic_matrix['ycb_K2']
        else:
            K = config.intrinsic_matrix['ycb_K1']

        with Image.open(os.path.join(self.root, item_name+'-color.png')) as ri:
            if self.add_noise:
                ri = self.trancolor(ri)
            rgb = np.array(ri)[:, :, :3]
        rnd_typ = 'syn' if 'syn' in item_name else 'real'
        cam_scale = meta['factor_depth'].astype(np.float32)[0][0]
        msk_dp = dpt_um > 1e-6

        if self.add_noise and rnd_typ == 'syn':
            rgb = self.rgb_add_noise(rgb)
            rgb, dpt_um = self.add_real_back(rgb, rgb_labels, dpt_um, msk_dp)
            if self.rng.rand() > 0.8:
                rgb = self.rgb_add_noise(rgb)

        dpt_um = bs_utils.fill_missing(dpt_um, cam_scale, 1)
        msk_dp = dpt_um > 1e-6

        dpt_mm = (dpt_um.copy()/10).astype(np.uint16)
        nrm_map = normalSpeed.depth_normal(
            dpt_mm, K[0][0], K[1][1], 5, 2000, 20, False
        )
        if self.debug:
            show_nrm_map = ((nrm_map + 1.0) * 127).astype(np.uint8)
            imshow("nrm_map", show_nrm_map)

        dpt_m = dpt_um.astype(np.float32) / cam_scale
        dpt_xyz_ori, rgb_pts_all, all_choose = self.dpt_2_pcld(dpt_m, rgb, 1.0, K)
        all_cld = dpt_xyz_ori.reshape(-1,3)
        rgb_pts_all = rgb_pts_all.reshape(-1,3)

        pred_cls_ids = np.unique(pred_rgb_labels)
        while(True): # get input cld, key point, choose idx, gt_kp_offset, gt_pose
            idx = np.random.randint(1, len(pred_cls_ids))
            cls_id = pred_cls_ids[idx]
            if cls_id in cls_id_lst:
                gt_idx = np.where(cls_id_lst == cls_id)[0][0]
                gt_r = meta['poses'][:, :, gt_idx][:, 0:3]
                gt_t = np.array(meta['poses'][:, :, gt_idx][:, 3:4].flatten()[:, None])
                RT = np.concatenate((gt_r, gt_t), axis=1) # gt_pose

                choose = np.arange(0, len(all_cld))
                msk_obj = (pred_rgb_labels.flatten() == cls_id)
                # msk_obj = (rgb_labels.flatten() == cls_id)
                pred_obj_pose = pred_pose[idx-1]

                if config.n_keypoints == 8:
                    kp_type = 'farthest'
                else:
                    kp_type = 'farthest{}'.format(config.n_keypoints)
                kps = bs_utils.get_kps(
                    self.cls_lst[cls_id-1], kp_type=kp_type, ds_type='ycb'
                ).copy()
                ctr = bs_utils.get_ctr(self.cls_lst[cls_id-1]).copy()
                mesh = bs_utils.get_pointxyzrgb(self.cls_lst[cls_id-1])[:, :3]
                r = pred_obj_pose[:, :3]
                t = pred_obj_pose[:, 3]
                r1 = np.dot(gt_r.T, r)
                t1 = np.dot(gt_t.T - t, r)
                kps_gt = np.dot(kps, r1) + t1
                ctr_gt = np.dot(ctr, r1) + t1
                pts_gt = np.dot(mesh, r1) + t1

                dpt_xyz = np.dot(all_cld - t, r).reshape(dpt_xyz_ori.shape)
                obj_cld = dpt_xyz.reshape(-1,3)[msk_obj]

                n_sample = 2048
                if obj_cld.shape[0] > 200:
                    kpcld_idx = np.random.permutation(obj_cld.shape[0])
                    if kpcld_idx.shape[0] < n_sample:
                        kpcld_idx = np.concatenate([kpcld_idx, np.random.randint(kpcld_idx.shape[0], size=n_sample - kpcld_idx.shape[0])])
                    kpcld_idx = kpcld_idx[:n_sample]
                    cld_kp = obj_cld[kpcld_idx]
                    ctr_gt_rp = np.repeat(ctr_gt, cld_kp.shape[0], axis=0)
                    ctr_ofst = ctr_gt_rp - cld_kp
                    kps_gt_rp = np.repeat(kps_gt[np.newaxis,:], cld_kp.shape[0], axis=0)
                    cld_kp_rp = np.repeat(cld_kp[:, np.newaxis, :], config.n_keypoints, axis=1)
                    kp_targ_ofst = kps_gt_rp - cld_kp_rp
                    ctr_targ_ofst = ctr_ofst[:, np.newaxis]
                    choose = choose[msk_obj][kpcld_idx]
                    break
        cld = cld_kp
        rgb_pt = rgb.reshape(-1, 3)[choose, :].astype(np.float32)
        nrm_pt = nrm_map[:, :, :3].reshape(-1, 3)[choose, :]
        labels_pt = labels.flatten()[choose]
        pred_labels_pt = pred_labels.flatten()[choose]
        choose = np.array([choose])
        cld_rgb_nrm = np.concatenate((cld, rgb_pt, nrm_pt), axis=1).transpose(1, 0)

        h, w = rgb_labels.shape
        dpt_6c = np.concatenate((dpt_xyz, nrm_map[:, :, :3]), axis=2).transpose(2, 0, 1)
        rgb = np.transpose(rgb, (2, 0, 1)) # hwc2chw
        xyz_lst = [dpt_xyz.transpose(2, 0, 1)] # c, h, w
        msk_lst = [dpt_xyz[2, :, :] > 1e-8]

        for i in range(3):
            scale = pow(2, i+1)
            nh, nw = h // pow(2, i+1), w // pow(2, i+1)
            ys, xs = np.mgrid[:nh, :nw]
            xyz_lst.append(xyz_lst[0][:, ys*scale, xs*scale])
            msk_lst.append(xyz_lst[-1][2, :, :] > 1e-8)
        sr2dptxyz = {
            pow(2, ii): item.reshape(3, -1).transpose(1, 0) for ii, item in enumerate(xyz_lst)
        }
        sr2msk = {
            pow(2, ii): item.reshape(-1) for ii, item in enumerate(msk_lst)
        }

        rgb_ds_sr = [4, 8, 8, 8]
        n_ds_layers = 4
        pcld_sub_s_r = [4, 4, 4, 4]
        inputs = {}
        # DownSample stage
        for i in range(n_ds_layers):
            nei_idx = DP.knn_search(
                cld[None, ...], cld[None, ...], 16
            ).astype(np.int32).squeeze(0)
            sub_pts = cld[:cld.shape[0] // pcld_sub_s_r[i], :]
            pool_i = nei_idx[:cld.shape[0] // pcld_sub_s_r[i], :]
            up_i = DP.knn_search(
                sub_pts[None, ...], cld[None, ...], 1
            ).astype(np.int32).squeeze(0)
            inputs['cld_xyz%d'%i] = cld.astype(np.float32).copy()
            inputs['cld_nei_idx%d'%i] = nei_idx.astype(np.int32).copy()
            inputs['cld_sub_idx%d'%i] = pool_i.astype(np.int32).copy()
            inputs['cld_interp_idx%d'%i] = up_i.astype(np.int32).copy()
            nei_r2p = DP.knn_search(
                sr2dptxyz[rgb_ds_sr[i]][None, ...], sub_pts[None, ...], 16
            ).astype(np.int32).squeeze(0)
            inputs['r2p_ds_nei_idx%d'%i] = nei_r2p.copy()
            nei_p2r = DP.knn_search(
                sub_pts[None, ...], sr2dptxyz[rgb_ds_sr[i]][None, ...], 1
            ).astype(np.int32).squeeze(0)
            inputs['p2r_ds_nei_idx%d'%i] = nei_p2r.copy()
            cld = sub_pts

        n_up_layers = 3
        rgb_up_sr = [4, 2, 2]
        for i in range(n_up_layers):
            r2p_nei = DP.knn_search(
                sr2dptxyz[rgb_up_sr[i]][None, ...],
                inputs['cld_xyz%d'%(n_ds_layers-i-1)][None, ...], 16
            ).astype(np.int32).squeeze(0)
            inputs['r2p_up_nei_idx%d'%i] = r2p_nei.copy()
            p2r_nei = DP.knn_search(
                inputs['cld_xyz%d'%(n_ds_layers-i-1)][None, ...],
                sr2dptxyz[rgb_up_sr[i]][None, ...], 1
            ).astype(np.int32).squeeze(0)
            inputs['p2r_up_nei_idx%d'%i] = p2r_nei.copy()

        show_rgb = rgb.transpose(1, 2, 0).copy()[:, :, ::-1]
        if self.debug:
            for ip, xyz in enumerate(xyz_lst):
                pcld = xyz.reshape(3, -1).transpose(1, 0)
                p2ds = bs_utils.project_p3d(pcld, cam_scale, K)
                srgb = bs_utils.paste_p2ds(show_rgb.copy(), p2ds, (0, 0, 255))
                p2ds = bs_utils.project_p3d(inputs['cld_xyz%d'%ip], cam_scale, K)
                srgb1 = bs_utils.paste_p2ds(show_rgb.copy(), p2ds, (0, 0, 255))
                imshow("rz_pcld_%d_rnd" % ip, srgb1)

        item_dict = dict(
            rgb=rgb.astype(np.uint8),  # [c, h, w]
            cld_rgb_nrm=cld_rgb_nrm.astype(np.float32),  # [9, npts]
            choose=choose.astype(np.int32),  # [1, npts]
            labels=labels_pt.astype(np.int32),  # [npts]
            pred_labels=pred_labels_pt.astype(np.int32),  # [npts]
            rgb_labels=rgb_labels.astype(np.int32),  # [h, w]
            # all_cld=all_cld.reshape(-1,3).astype(np.float32), #added by ourself
            pts_gt = pts_gt.astype(np.float32),
            RT=RT.astype(np.float32),
            kp_targ_ofst=kp_targ_ofst.astype(np.float32),
            ctr_targ_ofst=ctr_targ_ofst.astype(np.float32),
            # rgb_pt=rgb_pts_all.reshape(-1,3).astype(np.float32), #added by ourself
            # name=item_name,
        )
        item_dict.update(inputs)
        if self.debug:
            extra_d = dict(
                dpt_xyz_nrm=dpt_6c.astype(np.float32),  # [6, h, w]
                cam_scale=np.array([cam_scale]).astype(np.float32),
                K=K.astype(np.float32),
            )
            item_dict.update(extra_d)
            item_dict['normal_map'] = nrm_map[:, :, :3].astype(np.float32)
        return item_dict

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        if self.dataset_name == 'train':
            item_name = self.real_syn_gen()
            data = self.get_item(item_name)
            while data is None:
                item_name = self.real_syn_gen()
                data = self.get_item(item_name)
            return data
        else:
            item_name = self.all_lst[idx]
            return self.get_item(item_name)
