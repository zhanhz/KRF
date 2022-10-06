#!/usr/bin/env python3
from __future__ import (
    division,
    absolute_import,
    with_statement,
    print_function,
    unicode_literals,
)
import os
import tqdm
import cv2
import torch
import argparse
import torch.nn as nn
import numpy as np
import pickle as pkl
from common import Config, ConfigRandLA
from models.ffb6d import FFB6D
from datasets.ycb.ycb_dataset import Dataset as YCB_Dataset
from datasets.linemod.linemod_dataset import Dataset as LM_Dataset
from datasets.linemod.linemod_dataset_occ import Dataset as LM_OCC_Dataset
from utils.pvn3d_eval_utils_kpls import cal_frame_poses, cal_frame_poses_lm
from utils.basic_utils import Basic_Utils
from sklearn.neighbors import NearestNeighbors
try:
    from neupeak.utils.webcv2 import imshow, waitKey
except ImportError:
    from cv2 import imshow, waitKey
import warnings
warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser(description="Arg parser")
parser.add_argument(
    "-checkpoint", type=str, default=None, help="Checkpoint to eval"
)
parser.add_argument(
    "-dataset", type=str, default="linemod",
    help="Target dataset, ycb or linemod. (linemod as default)."
)
parser.add_argument(
    "-cls", type=str, default="ape",
    help="Target object to eval in LineMOD dataset. (ape, benchvise, cam, can," +
    "cat, driller, duck, eggbox, glue, holepuncher, iron, lamp, phone)"
)
parser.add_argument(
    "-gpu", type=str, default="0",
)
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
if args.dataset == "ycb":
    config = Config(ds_name=args.dataset)
else:
    config = Config(ds_name=args.dataset, cls_type=args.cls)
bs_utils = Basic_Utils(config)


def ensure_fd(fd):
    if not os.path.exists(fd):
        os.system('mkdir -p {}'.format(fd))


def load_checkpoint(model=None, optimizer=None, filename="checkpoint"):
    filename = "{}.pth.tar".format(filename)

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

def nearest_neighbor(src, dst):
    '''
    Find the nearest (Euclidean) neighbor in dst for each point in src
    Input:
        src: Nxm array of points
        dst: Nxm array of points
    Output:
        distances: Euclidean distances of the nearest neighbor
        indices: dst indices of the nearest neighbor
    '''

    assert src.shape == dst.shape

    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(dst)
    distances, indices = neigh.kneighbors(src, return_distance=True)
    return distances.ravel(), indices.ravel()

def get_all_msk(pcld, all_cld, mask):
    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(pcld)
    distances, indices = neigh.kneighbors(all_cld, return_distance=True)
    all_msk = mask[indices].reshape(-1)

    return all_msk

def cal_view_pred_pose(model, data, epoch=0, obj_id=-1):
    if args.dataset == 'ycb':
        root = 'datasets/ycb/YCB_Video_Dataset/add_data/'
    else:
        if(data['go'] == False): return
        root = 'datasets/linemod/Linemod_preprocessed/occ_data/{:0>2}/add_data/'.format(obj_id)
    model.eval()
    with torch.set_grad_enabled(False):
        cu_dt = {}
        for key in data.keys():
            if type(data[key]) != list:
                if data[key].dtype in [np.float32, np.uint8]:
                    cu_dt[key] = torch.from_numpy(data[key].astype(np.float32)).cuda()
                elif data[key].dtype in [np.int32, np.uint32]:
                    cu_dt[key] = torch.LongTensor(data[key].astype(np.int32)).cuda()
                elif data[key].dtype in [torch.uint8, torch.float32]:
                    cu_dt[key] = data[key].float().cuda()
                elif data[key].dtype in [torch.int32, torch.int16]:
                    cu_dt[key] = data[key].long().cuda()
        end_points = model(cu_dt)
        _, classes_rgbd = torch.max(end_points['pred_rgbd_segs'], 1)
        pcld = cu_dt['cld_rgb_nrm'][:, :3, :].permute(0, 2, 1).contiguous()
        all_cld = cu_dt['all_cld'][0]
        all_choose = cu_dt['all_choose'][0].cpu().numpy()
        all_cld = all_cld[all_choose]

        neigh = NearestNeighbors(n_neighbors=1)
        neigh.fit(pcld[0].cpu().numpy())
        distances, indices = neigh.kneighbors(all_cld.cpu().numpy(), return_distance=True)

        all_labels = classes_rgbd[0][indices].cpu().numpy()
        all_msk = np.zeros((480, 640), dtype="uint8")
        all_msk = all_msk.reshape(-1)
        all_msk[all_choose] = all_labels[:, 0]
        all_msk = all_msk.reshape((480, 640))

        if data['rnd_typ'][0] =='render':
            ensure_fd(os.path.join(root, 'render/pred_pose'))
            ensure_fd(os.path.join(root, 'render/pred_label'))   
            cv2.imwrite(root + 'render/pred_label/{}.png'.format(data['name'][0]), all_msk*255)
        elif data['rnd_typ'][0] =='fuse':
            ensure_fd(os.path.join(root, 'fuse/pred_pose'))
            ensure_fd(os.path.join(root, 'fuse/pred_label'))
            cv2.imwrite(root + 'fuse/pred_label/{}.png'.format(data['name'][0]), all_msk*255)
        else:
            ensure_fd(os.path.join(root, 'pred_label'))
            ensure_fd(os.path.join(root, 'pred_pose'))
            cv2.imwrite(root + 'pred_label/{}.png'.format(data['name'][0]), all_msk*255)

        if args.dataset == "ycb":
            pred_cls_ids, pred_pose_lst, _ = cal_frame_poses(
                pcld[0], classes_rgbd[0], end_points['pred_ctr_ofs'][0],
                end_points['pred_kp_ofs'][0], True, config.n_objects, True,
                None, None
            )

            from utils.meanshift_pytorch import MeanShiftTorch
            pred_ctr = pcld[0] - end_points['pred_ctr_ofs'][0][0] 
            mask = classes_rgbd[0]
            pred_cls_ids = np.unique(mask[mask > 0].contiguous().cpu().numpy())
            new_msk = torch.zeros_like(mask).cpu()
            for icls, cls_id in enumerate(pred_cls_ids):
                cls_msk = torch.where(mask == cls_id)[0]
                ms = MeanShiftTorch(bandwidth=0.02)
                ctr, ctr_labels = ms.fit(pred_ctr[cls_msk, :])
                new_msk[cls_msk[ctr_labels]] = cls_id.item()
                

            all_labels = new_msk[indices].cpu().numpy()
            all_msk = np.zeros((480, 640), dtype="uint8")
            all_msk = all_msk.reshape(-1)
            all_msk[all_choose] = all_labels[:, 0]
            all_msk = all_msk.reshape((480, 640))
            ensure_fd(os.path.join(root, data['name'][0][:-6]))
            cv2.imwrite(root + '{}-pred_label.png'.format(data['name'][0]), all_msk)
            np.save(root + '{}-pred_pose.npy'.format(data['name'][0]), (pred_pose_lst))
        else:
            pred_pose_lst = cal_frame_poses_lm(
                pcld[0], classes_rgbd[0], end_points['pred_ctr_ofs'][0],
                end_points['pred_kp_ofs'][0], True, config.n_objects, False, obj_id
            )
            pred_cls_ids = np.array([[1]])

            if data['rnd_typ'][0] =='render':
                np.save(root + 'render/pred_pose/{}.npy'.format(data['name'][0]), (pred_pose_lst))
            elif data['rnd_typ'][0] =='fuse':
                np.save(root + 'fuse/pred_pose/{}.npy'.format(data['name'][0]), (pred_pose_lst))
            else:
                np.save(root + 'pred_pose/{}.npy'.format(data['name'][0]), (pred_pose_lst))
def main():
    if args.dataset == "ycb":
        test_ds = YCB_Dataset('test')
        obj_id = -1
    else:
        # test_ds = LM_Dataset('test', cls_type=args.cls)
        test_ds = LM_OCC_Dataset('test', cls_type=args.cls)
        obj_id = config.lm_obj_dict[args.cls]
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=config.test_mini_batch_size, shuffle=False,
        num_workers=8
    )

    rndla_cfg = ConfigRandLA
    model = FFB6D(
        n_classes=config.n_objects, n_pts=config.n_sample_points, rndla_cfg=rndla_cfg,
        n_kps=config.n_keypoints
    )
    model.cuda()

    # load status from checkpoint
    if args.checkpoint is not None:
        load_checkpoint(
            model, None, filename=args.checkpoint[:-8]
        )
    for i, data in tqdm.tqdm(
        enumerate(test_loader), leave=False, desc="val"
    ):
        cal_view_pred_pose(model, data, epoch=i, obj_id=obj_id)


if __name__ == "__main__":
    main()

# vim: ts=4 sw=4 sts=4 expandtab
