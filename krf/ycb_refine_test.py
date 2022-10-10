#!/usr/bin/env python3
import os
import argparse
parser = argparse.ArgumentParser(description="Arg parser")
parser.add_argument(
    "-ckpt", "--checkpoint", type=str, default='train_log/ycb/checkpoints/FFB6D_REFINE_pcn.pth.tar', help="Checkpoint to eval"
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


import torch
import time
import os.path
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from common import Config, ConfigRandLA
from models.ffb6d_refine_pcn import FFB6D_REFINE
import pickle as pkl
import datasets.ycb.ycb_refine_8kps_test_dataset as dataset_desc
from utils.ycb_refine_eval import TorchEval
from utils.basic_utils import Basic_Utils
from utils.meanshift_pytorch import MeanShiftTorch
from sklearn.cluster import MeanShift
import scipy.io as scio
from utils.icp.icp import fix_R_icprgb, my_icprgb_torch
from models.loss import ChamferDistance as cd_loss
cd = cd_loss()
config = Config(ds_name='ycb')
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

def calc_pcn_icp_pose(model, ds, item_name, cls_lst, cls_id, pred_obj_pose, n_iter=1):
    use_pcld = args.use_pcld
    use_rgb = args.use_rgb
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
        mesh_kps = bs_utils.get_kps(cls_lst[cls_id-1]).copy()
        mesh_ctr = bs_utils.get_ctr(cls_lst[cls_id-1]).reshape(1, 3)
        mesh_kpc = np.concatenate((mesh_kps, mesh_ctr), axis=0)
        pred_obj_pose_cu = torch.from_numpy(pred_obj_pose.astype(np.float32)).cuda()
        r = pred_obj_pose_cu[:, :3]
        t = pred_obj_pose_cu[:, 3]
        rc = pred_obj_pose[:, :3]
        tc = pred_obj_pose[:, 3]
        ori_kpc = np.dot(mesh_kpc, rc.T) + tc  #original kps position
        ori_kps = ori_kpc[:-1]
        refined_kps = torch.from_numpy(ori_kpc).clone().cuda()
        mesh_pts = bs_utils.get_pointxyzrgb_cuda(cls_lst[cls_id-1])[:, :3].clone()
        mesh_pts_rgb = bs_utils.get_pointxyzrgb_cuda(cls_lst[cls_id-1]).clone()
        icp_pose = pred_obj_pose_cu
        kps = ori_kpc.copy()
        datum = ds.get_item(item_name, cls_id, pred_obj_pose)
        cu_dt={}
        if datum is not None:
            for key in datum.keys():
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
            #icp
            if not use_rgb:
                if cld.shape[0] > 1500:
                    pcld = torch.cat([pcld, torch.zeros((pcld.shape[0], 4)).cuda()], 1)
                    cld = torch.cat([cld, torch.zeros((cld.shape[0], 4)).cuda()], 1)
                    mesh_pts = torch.cat([mesh_pts, torch.zeros((mesh_pts.shape[0], 4)).cuda()], 1)
                    if use_pcld and dist.item() < 0.1: cld = torch.cat([cld, pcld], axis=0)
                    icp_pose, _, _ = my_icprgb_torch(mesh_pts, cld, icp_pose, max_iterations=200, tolerance=1e-6)
                    icp_pose = icp_pose[:3, :]
#=====================================================================================
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
                        kpsnb_msk = ikp_dist < config.ycb_r_lst[cls_id-1] * 0.8
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

    return icp_pose.contiguous().cpu().numpy(), refined_kps.contiguous().cpu().numpy()[:-1]

def eval_metric(
    cls_ids, pred_pose_lst, refine_pose_lst, pred_cls_ids, RTs, gt_kps, refine_kps_lst, cls_lst
):# input: gt, pred, refine pose; output: pred, refine result
    n_cls = config.n_classes
    cls_add_dis = [list() for i in range(n_cls)]
    cls_adds_dis = [list() for i in range(n_cls)]
    cls_refine_add_dis = [list() for i in range(n_cls)]
    cls_refine_adds_dis = [list() for i in range(n_cls)]

    cls_kp_err = [list() for i in range(n_cls)]
    for icls, cls_id in enumerate(cls_ids):
        if cls_id == 0:
            break

        gt_kp = gt_kps[icls]

        cls_idx = np.where(pred_cls_ids == cls_id)[0]
        if len(cls_idx) == 0:
            pred_RT = torch.zeros(3, 4).cuda()
            pred_RT_r = pred_RT.clone()
            pred_kp = np.zeros(gt_kp.shape)
        else:
            pred_RT = pred_pose_lst[cls_idx[0]]
            pred_RT = torch.from_numpy(pred_RT.astype(np.float32)).cuda()

            pred_RT_r = refine_pose_lst[cls_idx[0]]
            pred_kp = refine_kps_lst[cls_idx[0]]
            pred_RT_r = torch.from_numpy(pred_RT_r.astype(np.float32)).cuda()

        kp_err = np.linalg.norm(gt_kp-pred_kp, axis=1).mean()
        cls_kp_err[cls_id].append(kp_err)
        gt_RT = torch.from_numpy(RTs[icls].astype(np.float32)).cuda()
        mesh_pts = bs_utils.get_pointxyz_cuda(cls_lst[cls_id-1]).clone()

        add = bs_utils.cal_add_cuda(pred_RT, gt_RT, mesh_pts)
        adds = bs_utils.cal_adds_cuda(pred_RT, gt_RT, mesh_pts)
        cls_add_dis[cls_id].append(add.item())
        cls_adds_dis[cls_id].append(adds.item())
        cls_add_dis[0].append(add.item())
        cls_adds_dis[0].append(adds.item())

        add_r = bs_utils.cal_add_cuda(pred_RT_r, gt_RT, mesh_pts)
        adds_r = bs_utils.cal_adds_cuda(pred_RT_r, gt_RT, mesh_pts)
        cls_refine_add_dis[cls_id].append(add_r.item())
        cls_refine_adds_dis[cls_id].append(adds_r.item())
        cls_refine_add_dis[0].append(add_r.item())
        cls_refine_adds_dis[0].append(adds_r.item())
        print('%.4f %.4f %.4f %.4f'%(add.item(), add_r.item(), adds.item(), adds_r.item()))
    return (cls_add_dis, cls_adds_dis, cls_refine_add_dis, cls_refine_adds_dis, cls_kp_err)
    
    
def main():
    teval = TorchEval()
    root = config.ycb_root
    pred_root = os.path.join(root, 'add_data')
    all_lst = bs_utils.read_lines('datasets/ycb/dataset_config/test_data_list.txt')
    ds = dataset_desc.Dataset('test')

    cls_lst = config.ycb_cls_lst
    print('test size: {}'.format(len(all_lst)))

    rndla_cfg = ConfigRandLA
    model = FFB6D_REFINE(
        n_classes=config.n_objects, n_pts=config.n_sample_points, rndla_cfg=rndla_cfg,
        n_kps=config.n_keypoints
    )
    model.cuda()
    load_checkpoint(
        model, None, filename=args.checkpoint
    )
    model.eval()
    with torch.no_grad():
        all_start = time.time()
        for idx in range(len(all_lst)): #each image
            print(idx)
            start = time.time()
            item_name = all_lst[idx]
            print(item_name)
            with Image.open(os.path.join(pred_root, item_name+ '-pred_label.png')) as pli:
                pred_labels = np.array(pli)
            pred_rgb_labels = pred_labels.copy()
            pred_cls_ids = np.unique(pred_rgb_labels[pred_rgb_labels>0])
            pred_pose = np.load(os.path.join(pred_root, item_name+'-pred_pose.npy'))
            refine_pose_lst = pred_pose.copy()
            refine_kps_lst = np.zeros((refine_pose_lst.shape[0], config.n_keypoints, 3))
            meta = scio.loadmat(os.path.join(root, item_name+'-meta.mat'))
            cls_id_lst = meta['cls_indexes'].flatten().astype(np.uint32)
            
            RTs = np.zeros((len(cls_id_lst), 3, 4))
            gt_kps = np.zeros((len(cls_id_lst), config.n_keypoints, 3))
            for i, cls_id in enumerate(cls_id_lst):
                r = meta['poses'][:, :, i][:, 0:3]
                t = np.array(meta['poses'][:, :, i][:, 3:4].flatten()[:, None])
                RT = np.concatenate((r, t), axis=1)
                RTs[i] = RT # gt_pose  

                kps = bs_utils.get_kps(cls_lst[cls_id-1]).copy()
                kps = np.dot(kps, r.T) + t[:, 0]
                gt_kps[i] = kps

            cls_id_lst = meta['cls_indexes'].flatten().astype(np.uint32)
            for i, cls_id in enumerate(cls_id_lst): #each object
                if cls_id in pred_cls_ids:
                    pred_id = np.where(pred_cls_ids == cls_id)[0][0]
                    pred_obj_pose = pred_pose[pred_id]
                    refine_pose, refine_kps = calc_pcn_icp_pose(model, ds, item_name, cls_lst, cls_id, pred_obj_pose, n_iter=10)# each keypoint

                    refine_pose_lst[pred_id] = refine_pose
                    refine_kps_lst[pred_id] = refine_kps
            cls_add_dis, cls_adds_dis, cls_refine_add_dis, cls_refine_adds_dis, cls_kp_err = eval_metric(
                cls_id_lst, pred_pose, refine_pose_lst, pred_cls_ids, RTs, gt_kps, refine_kps_lst, cls_lst
                )
            teval.push(cls_add_dis, cls_adds_dis, cls_refine_add_dis, cls_refine_adds_dis, pred_cls_ids, pred_pose, cls_kp_err)
            # print(time.time()-start)
        teval.cal_auc()
        print(time.time()-all_start)
        
if __name__ == "__main__":
    main()
# vim: ts=4 sw=4 sts=4 expandtab
