#!/usr/bin/env python3
import os
import argparse
parser = argparse.ArgumentParser(description="Arg parser")
parser.add_argument(
    "-ckpt", "--checkpoint", type=str, default='train_log/linemod/checkpoints/', help="Checkpoint folder"
)
parser.add_argument(
    "-cls", type=str, default="ape",
    help="Target object to eval in LineMOD dataset. (ape, benchvise, cam, can," +
    "cat, driller, duck, eggbox, glue, holepuncher, iron, lamp, phone)"
)
parser.add_argument(
    "-gpu", type=str, default="0",
)
parser.add_argument(
    "-use_pcld", action='store_true', default=False
)
parser.add_argument(
    "-use_rgb", action='store_true', default=False
)
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

import cv2
import torch
import time
import os.path
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from common import Config, ConfigRandLA
# from models.ffb6d_refine import FFB6D_REFINE
from models.ffb6d_refine_pcn import FFB6D_REFINE
# from models.ffb6d_refine_8kps import FFB6D_REFINE
# from models.pvn3d import PVN3D
import pickle as pkl
# import ycb_refine_test_dataset as dataset_desc
import datasets.linemod.linemod_testocc_dataset as dataset_desc
from utils.lm_refine_eval import TorchEval
from utils.basic_utils import Basic_Utils
from utils.meanshift_pytorch import MeanShiftTorch
from sklearn.cluster import MeanShift
import scipy.io as scio
import scipy.misc
import normalSpeed
import yaml
from utils.icp.icp import my_icprgb, my_icp_torch, my_icprgb_torch, fix_R_icp, fix_R_icprgb

config = Config(ds_name='linemod')
bs_utils = Basic_Utils(config)
def get_cld_bigest_clus_cpu(p3ds):
    n_clus_jobs = 8
    ms = MeanShift(
        bandwidth=0.06, bin_seeding=True, n_jobs=n_clus_jobs
    )
    ms.fit(p3ds)
    clus_labels = ms.labels_
    bg_clus = p3ds[np.where(clus_labels == 0)[0], :]
    return bg_clus

def best_fit_transform(A, B):
    '''
    Calculates the least-squares best-fit transform that maps corresponding points A to B in m spatial dimensions
    Input:
        A: Nxm numpy array of corresponding points, usually points on mdl
        B: Nxm numpy array of corresponding points, usually points on camera axis
    Returns:
    T: (m+1)x(m+1) homogeneous transformation matrix that maps A on to B
    R: mxm rotation matrix
    t: mx1 translation vector
    '''

    assert A.shape == B.shape
    # get number of dimensions
    m = A.shape[1]
    # translate points to their centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B
    # rotation matirx
    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)
    # special reflection case
    if np.linalg.det(R) < 0:
        Vt[m-1, :] *= -1
        R = np.dot(Vt.T, U.T)
    # translation
    t = centroid_B.T - np.dot(R, centroid_A.T)
    T = np.zeros((3, 4))
    T[:, :3] = R
    T[:, 3] = t
    return T

def load_checkpoint(model=None, optimizer=None, filename="checkpoint"):
    # filename = "{}.pth.tar".format(filename)

    assert os.path.isfile(filename), "==> Checkpoint '{}' not found".format(filename)
    print("==> Loading from checkpoint '{}'".format(filename))
    try:
        checkpoint = torch.load(filename)
    except Exception:
        checkpoint = pkl.load(open(filename, "rb"))
    epoch = checkpoint.get("epoch", 0)
    it = checkpoint.get("it", 0.0)
    best_prec = checkpoint.get("best_prec", None)
    if model is not None and checkpoint["model_state"] is not None:
        ck_st = checkpoint['model_state']
        if 'module' in list(ck_st.keys())[0]:
            tmp_ck_st = {}
            for k, v in ck_st.items():
                tmp_ck_st[k.replace("module.", "")] = v
            ck_st = tmp_ck_st
        model.load_state_dict(ck_st)
    if optimizer is not None and checkpoint["optimizer_state"] is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    print("==> Done")
    return it, epoch, best_prec

def calc_pcn_icp_pose(model, ds, item_name, obj_id, pred_obj_pose, n_iter=1):
    def get_cld_bigest_clus_torch(p3ds, cld_rgb):
        n_clus_jobs = 8
        ms = MeanShiftTorch(
            bandwidth=0.06
        )
        _, clus_labels = ms.fit(p3ds)
        bg_clus = p3ds[clus_labels, :]
        if (cld_rgb == None):
            return bg_clus
        bg_clus_rgb = cld_rgb[clus_labels, :]
        return bg_clus,bg_clus_rgb
    with torch.set_grad_enabled(False):
        cu_dt = {}
        mesh_kps = bs_utils.get_kps(obj_id, ds_type='linemod').copy()
        mesh_ctr = bs_utils.get_ctr(obj_id, ds_type='linemod').reshape(1, 3)
        mesh_kpc = np.concatenate((mesh_kps, mesh_ctr), axis=0)
        pred_obj_pose_cu = torch.from_numpy(pred_obj_pose.astype(np.float32)).cuda()
        r = pred_obj_pose_cu[:, :3]
        t = pred_obj_pose_cu[:, 3]
        rc = pred_obj_pose[:, :3]
        tc = pred_obj_pose[:, 3]
        ori_kpc = np.dot(mesh_kpc, rc.T) + tc  #original kps position
        ori_kps = ori_kpc[:-1]
        refined_kps = torch.from_numpy(ori_kpc).clone().cuda()
        mesh_pts = bs_utils.get_pointxyzrgb_cuda(obj_id, ds_type='linemod')[:, :3].clone()
        mesh_pts_rgb = bs_utils.get_pointxyzrgb_cuda(obj_id, ds_type='linemod').clone()
        icp_pose = pred_obj_pose_cu
        
        kps = ori_kpc.copy()
        datum = ds.get_item(item_name, pred_obj_pose)
        cu_dt={}
        if datum is not None:
            for key in datum.keys():
                if type(datum[key]) != list:
                    if datum[key].dtype in [np.float32, np.uint8]:
                        cu_dt[key] = torch.from_numpy(datum[key].astype(np.float32)).cuda()
                    elif datum[key].dtype in [np.int32, np.uint32]:
                        cu_dt[key] = torch.LongTensor(datum[key].astype(np.int32)).cuda()
                    elif datum[key].dtype in [torch.uint8, torch.float32]:
                        cu_dt[key] = datum[key].float().cuda()
                    elif datum[key].dtype in [torch.int32, torch.int16]:
                        cu_dt[key] = datum[key].long().cuda()
            end_points = model(cu_dt)
            pcld = end_points['detail'][0].T
            gt_pts = cu_dt['all_cld'][0]
            rgb_pts = cu_dt['rgb_pt'][0]

            cld_len = len(gt_pts)
            if cld_len > 3000:
                c_mask = np.zeros(cld_len, dtype=int)
                c_mask[:3000] = 1
                np.random.shuffle(c_mask)
                cld = gt_pts[c_mask.nonzero()[0], :]
                rgb = rgb_pts[c_mask.nonzero()[0], :] / 255.0
            else: 
                cld = gt_pts
                rgb = rgb_pts / 255.0
            cld_rgb = torch.cat([cld, rgb], axis=-1)
            cld, cld_rgb = get_cld_bigest_clus_torch(cld, cld_rgb)

            pcld = torch.mm(pcld, r.T) + t
            pctr = torch.mean(pcld, 0)
            gctr = torch.mean(cld, 0)
            dist = torch.norm(pctr - gctr)

            use_pcld = True
            use_rgb = True

            #icp
            if not use_rgb:
                if cld.shape[0] > 1500:
                    pcld = torch.cat([pcld, torch.zeros((pcld.shape[0], 4)).cuda()], 1)
                    cld = torch.cat([cld, torch.zeros((cld.shape[0], 4)).cuda()], 1)
                    mesh_pts = torch.cat([mesh_pts, torch.zeros((mesh_pts.shape[0], 4)).cuda()], 1)
                    if use_pcld and dist.item() < 0.1: cld = torch.cat([cld, pcld], axis=0)
                    icp_pose, _, _ = my_icprgb_torch(mesh_pts, cld, icp_pose, max_iterations=200, tolerance=1e-6)
                    icp_pose = icp_pose[:3, :]
            #cikp
            else:
                if use_pcld and dist.item() < 0.1: cld = torch.cat([cld, pcld], axis=0)
                cld_rgb = torch.cat([cld_rgb, torch.ones(cld_rgb.shape[0],1).cuda()], 1)
                pcld = torch.cat([pcld, torch.zeros((pcld.shape[0], 4)).cuda()], 1)
                mesh_pts_rgb = torch.cat([mesh_pts_rgb, torch.ones(mesh_pts_rgb.shape[0],1).cuda()], 1)
                n_pts = cld.shape[0]
                n_kp = ori_kps.shape[0]
                mesh_kps_cuda = torch.from_numpy(mesh_kpc).cuda()
                if use_pcld and dist.item() < 0.1: cld_rgb = torch.cat([cld_rgb, pcld], axis=0)

                prev_err = 0
                for i in range(n_iter):
                    r = icp_pose[:, :3].cpu().numpy()
                    t = icp_pose[:, 3].cpu().numpy()
                    ori_kpc = np.dot(mesh_kpc, r.T) + t  #original kps position
                    ori_kps = ori_kpc[:-1]
                    kps_rp = torch.from_numpy(ori_kps).view(n_kp, 1, 3).repeat(1, n_pts, 1).cuda()
                    cld_rp = cld.view(1, n_pts, 3).repeat(n_kp, 1, 1)
                    kps_dist = torch.norm((kps_rp - cld_rp), dim=2)
                    refined_kps = torch.from_numpy(ori_kpc).clone().cuda()
                    mean_err = torch.mean(kps_dist)
                    if torch.abs(prev_err - mean_err) < 1e-6: 
                        break
                    prev_err = mean_err
                    for ikp in range(n_kp):
                        kp_xyz = mesh_kps_cuda[ikp,:]
                        ikp_dist = kps_dist[ikp]
                        kpsnb_msk = ikp_dist < config.lm_r_lst[obj_id]['diameter'] * 0.7 * 0.0005
                        cld_kp = cld[kpsnb_msk]
                        cld_rgb_kp = cld_rgb[kpsnb_msk]
                        if cld_kp.shape[0] < 100: continue              
                        kp_pose, _, _ = fix_R_icprgb(mesh_pts_rgb, cld_rgb_kp, icp_pose)

                        kp_pose = kp_pose[:3, :]
                        refine_kp = torch.mm(kp_xyz.unsqueeze(0), kp_pose[:, :3].T ).view(3) + kp_pose[:, 3]
                        refined_kps[ikp] = refine_kp

                    icp_pose = bs_utils.best_fit_transform(
                        mesh_kpc.astype(np.float32),
                        refined_kps.contiguous().cpu().numpy()
                    )
                    icp_pose = icp_pose[:3, :]
                    icp_pose = torch.tensor(icp_pose.astype(np.float32)).cuda()
    return icp_pose.contiguous().cpu().numpy()

def eval_metric(
    pred_pose_lst, refine_pose_lst, obj_id, RTs
):
    n_cls = config.n_classes
    cls_add_dis = [list() for i in range(n_cls)]
    cls_adds_dis = [list() for i in range(n_cls)]
    cls_refine_add_dis = [list() for i in range(n_cls)]
    cls_refine_adds_dis = [list() for i in range(n_cls)]

    pred_RT = pred_pose_lst[0]
    pred_RT = torch.from_numpy(pred_RT.astype(np.float32)).cuda()

    pred_RT_r = refine_pose_lst[0]
    pred_RT_r = torch.from_numpy(pred_RT_r.astype(np.float32)).cuda()

    gt_RT = torch.from_numpy(RTs[0].astype(np.float32)).cuda()
    mesh_pts = bs_utils.get_pointxyz_cuda(obj_id, ds_type="linemod").clone()
    add = bs_utils.cal_add_cuda(pred_RT, gt_RT, mesh_pts)
    adds = bs_utils.cal_adds_cuda(pred_RT, gt_RT, mesh_pts)
    cls_add_dis[1].append(add.item())
    cls_adds_dis[1].append(adds.item())
    cls_add_dis[0].append(add.item())
    cls_adds_dis[0].append(adds.item()) 

    add_r = bs_utils.cal_add_cuda(pred_RT_r, gt_RT, mesh_pts)
    adds_r = bs_utils.cal_adds_cuda(pred_RT_r, gt_RT, mesh_pts)
    cls_refine_add_dis[1].append(add_r.item())
    cls_refine_adds_dis[1].append(adds_r.item())
    cls_refine_add_dis[0].append(add_r.item())
    cls_refine_adds_dis[0].append(adds_r.item())
    return (cls_add_dis, cls_adds_dis, cls_refine_add_dis, cls_refine_adds_dis)
    
    
def main():
    cls_type = args.cls
    obj_dict = config.lm_obj_dict
    obj_id = obj_dict[cls_type]

    teval = TorchEval()
    root = os.path.join(config.lm_root, 'Linemod_preprocessed')
    occ_root = os.path.join(root, "data/02/")
    cls_root = os.path.join(root, 'occ_data/%02d' %obj_id)
    all_lst = bs_utils.read_lines(os.path.join(occ_root, 'test_occ.txt'))
    ds = dataset_desc.Dataset('test', cls_type)

    print('test size: {}'.format(len(all_lst)))

    rndla_cfg = ConfigRandLA
    model = FFB6D_REFINE(
        n_classes=config.n_objects, n_pts=config.n_sample_points, rndla_cfg=rndla_cfg,
        n_kps=config.n_keypoints
    )
    model.cuda()
    load_checkpoint(
        model, None, filename= args.checkpoint+'{}/FFB6D_{}_REFINE_best.pth.tar'.format(cls_type, cls_type)
    )
    model.eval()
    all_start = time.time()
    meta_file = open(os.path.join(occ_root, 'gt.yml'), "r")
    meta_lst = yaml.load(meta_file)
    for idx in range(len(all_lst)): #each image
        start = time.time()
        item_name = all_lst[idx]

        meta = meta_lst[int(item_name)]
        mm = None
        for i in range(0, len(meta)):
            if meta[i]['obj_id'] == obj_id:
                mm = meta[i]
                break
        if mm == None: continue
        meta = mm

        with Image.open(os.path.join(cls_root, "add_data/pred_label/{}.png".format(item_name))) as pli:
            pred_labels = np.array(pli)
        pred_rgb_labels = pred_labels.copy()
        if(np.where(pred_rgb_labels.flatten() > 0)[0].shape[0] == 0): continue
        pred_pose = np.load(os.path.join(cls_root, "add_data/pred_pose/{}.npy".format(item_name)))
        refine_pose_lst = pred_pose.copy()

        RTs = np.zeros((1, 3, 4))
        gt_kps = np.zeros((1, config.n_keypoints, 3))

        R = np.resize(np.array(meta['cam_R_m2c']), (3, 3))
        T = np.array(meta['cam_t_m2c']) / 1000.0
        RT = np.concatenate((R, T[:, None]), axis=1)
        RTs[0] = RT # gt_pose  

        kps = bs_utils.get_kps(obj_id, ds_type='linemod').copy()
        kps = np.dot(kps, R.T) + T
        gt_kps[0] = kps

        pred_obj_pose = pred_pose[0]        
        refine_pose = calc_pcn_icp_pose(model, ds, item_name, obj_id, pred_obj_pose, n_iter=10)
        refine_pose_lst[0] = refine_pose
        
        
        cls_add_dis, cls_adds_dis, cls_refine_add_dis, cls_refine_adds_dis = eval_metric(
            pred_pose, refine_pose_lst, obj_id, RTs
            )
        teval.push(cls_add_dis, cls_adds_dis, cls_refine_add_dis, cls_refine_adds_dis, pred_pose)
        print(time.time()-start)
    teval.cal_auc(obj_id)
    print(time.time()-all_start)

if __name__ == "__main__":
    main()
# vim: ts=4 sw=4 sts=4 expandtab
