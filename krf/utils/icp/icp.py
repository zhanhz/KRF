import numpy as np
import cv2
import torch
import os
from sklearn.neighbors import NearestNeighbors
from knn_cuda import KNN

def fix_R_transform(A, B):
    assert A.size() == B.size()
    A = A.detach().cpu().numpy()
    B = B.detach().cpu().numpy()
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    t = centroid_B.T - centroid_A.T
    T = np.identity(4)
    T[:3, 3] = t
    return torch.from_numpy(T.astype(np.float32)).cuda()

def fix_R_icp(A, B, init_pose=None, max_iterations=20, tolerance=0.001):
    # get number of dimensions
    m = 3

    # make points homogeneous, copy them to maintain the originals
    src = torch.ones((4,A.shape[0])).cuda()
    dst = torch.ones((4,B.shape[0])).cuda()
    src[:3,:] = A[:, :3].T.clone()
    dst[:3,:] = B[:, :3].T.clone()

    pose = torch.eye(4).cuda()
    init_R = torch.eye(4).cuda()

    if init_pose is not None:
        pose[:3, :] = init_pose
        init_R[:3, :3] = init_pose[:, :3]
        A_R = torch.mm(init_R, src)
        src = torch.mm(pose, src)


    init_distances, _ = my_nearest_neighbor_cuda(dst[:m,:].T, src[:m,:].T, eu_dist=True)

    distances, indices = my_nearest_neighbor_cuda(dst[:m,:].T, src[:m,:].T, eu_dist=True)
    T = fix_R_transform(src[:m,indices].T, dst[:m,:].T)
    src = torch.mm(T, src)

    T = fix_R_transform(A_R[:3,:].T, src[:3,:].T)
    T[:3, :3] = init_R[:3, :3]

    return T, distances, init_distances

def fix_R_icprgb(A, B, init_pose=None, max_iterations=20, tolerance=0.001):
    # get number of dimensions
    m = 3

    # make points homogeneous, copy them to maintain the originals
    src_rgb = (torch.clone(A[:, 3:]) * 0.007).cuda()
    dst_rgb = (torch.clone(B[:, 3:]) * 0.007).cuda()
    src = torch.ones((4,A.shape[0])).cuda()
    dst = torch.ones((4,B.shape[0])).cuda()
    src_rgb_t = torch.clone(A[:, 3:]).cuda()
    dst_rgb_t = torch.clone(B[:, 3:]).cuda()
    src[:3,:] = A[:, :3].T.clone()
    dst[:3,:] = B[:, :3].T.clone()

    pose = torch.eye(4).cuda()
    init_R = torch.eye(4).cuda()

    if init_pose is not None:
        pose[:3, :] = init_pose
        init_R[:3, :3] = init_pose[:, :3]
        A_R = torch.mm(init_R, src)
        src = torch.mm(pose, src)
        init_src = src.clone()


    prev_error = 0
    init_distances, _ = my_nearest_neighbor_cuda(torch.cat([dst[:3,:].T, dst_rgb], -1), torch.cat([src[:3,:].T, src_rgb], -1))

    distances, indices = my_nearest_neighbor_cuda(dst[:m,:].T, src[:m,:].T)
    T = fix_R_transform(src[:m,indices].T, dst[:m,:].T)
    src = torch.mm(T, src)

    distances, indices = my_nearest_neighbor_cuda(torch.cat([dst[:3,:].T, dst_rgb], -1), torch.cat([src[:3,:].T, src_rgb], -1))
    T = fix_R_transform(src[:m,indices].T, dst[:m,:].T)
    src = torch.mm(T, src)
    # T,_,_ = best_fit_transform_torch(A[:, :3], src[:3,:].T)
    if np.mean(distances) < np.mean(init_distances):
        T = fix_R_transform(A_R[:3,:].T, src[:3,:].T)
        T[:3, :3] = init_R[:3, :3]
    else:
        T = pose
    return T, distances, init_distances




def best_fit_transform_torch(A, B):
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
    assert A.size() == B.size()
    A = A.detach().cpu().numpy()
    B = B.detach().cpu().numpy()
    T, R, t = best_fit_transform(A, B)
    T = torch.from_numpy(T.astype(np.float32)).cuda()
    return T, R, t


def best_fit_transform(A, B):
    '''
    Calculates the least-squares best-fit transform that maps corresponding points A to B in m spatial dimensions
    Input:
      A: Nxm numpy array of corresponding points
      B: Nxm numpy array of corresponding points
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

    # rotation matrix
    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    # special reflection case
    if np.linalg.det(R) < 0:
       Vt[m-1,:] *= -1
       R = np.dot(Vt.T, U.T)

    # translation
    t = centroid_B.T - np.dot(R,centroid_A.T)

    # homogeneous transformation
    T = np.identity(m+1)
    T[:m, :m] = R
    T[:m, m] = t
    # print(T.dtype)

    return T, R, t

def my_nearest_neighbor_cuda(src, dst, eu_dist=False):
    m = src.shape[-1]
    knn = KNN(k = 1, transpose_mode=True, eu_dist=eu_dist)
    ref = dst.reshape((1, -1, m))
    query = src.reshape((1, -1, m))
    dist, idx = knn(ref, query)
    return dist.cpu().numpy().flatten(), idx.cpu().numpy().flatten()

def my_icp_torch(A, B, init_pose=None, max_iterations=20, tolerance=0.001):
    m = A.shape[1]
    src = torch.ones((m+1, A.shape[0])).cuda()
    dst = torch.ones((m+1, B.shape[0])).cuda()
    src[:m, :] = A.T.clone()
    pose = torch.eye(4).cuda()
    dst[:m, :] = B.T.clone()

    if init_pose is not None:
        pose[:3, :] = init_pose
        src = torch.mm(pose, src)
        init_src = src.clone()


    prev_error = 0
    init_distances, _ = my_nearest_neighbor_cuda(dst[:m,:].T, src[:m,:].T, eu_dist=True)
    for i in range(max_iterations):
        distances, indices = my_nearest_neighbor_cuda(dst[:m,:].T, src[:m,:].T, eu_dist=True)
        T,_,_ = best_fit_transform_torch(src[:m,indices].T, dst[:m,:].T)
        src = torch.mm(T, src)

        mean_error = np.mean(distances)

        if np.abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error
        # print(mean_error)
    
    if mean_error < np.mean(init_distances):
        T,_,_ = best_fit_transform_torch(A, src[:m,:].T)
    else: 
        T = pose


    return T, distances, init_distances

def my_icprgb_torch(A, B, init_pose=None, max_iterations=50, tolerance=0.00001):
    '''
    The Iterative Closest Point method: finds best-fit transform that maps points A on to points B
    Input:
        A: Nxm numpy array of source mD points
        B: Nxm numpy array of destination mD point
        init_pose: (m+1)x(m+1) homogeneous transformation
        max_iterations: exit algorithm after max_iterations
        tolerance: convergence criteria
    Output:
        T: final homogeneous transformation that maps A on to B
        distances: Euclidean distances (errors) of the nearest neighbor
        i: number of iterations to converge
    '''

    # get number of dimensions
    m = A.shape[1]

    # make points homogeneous, copy them to maintain the originals
    src_rgb = (torch.clone(A[:, 3:]) * 0.002).cuda()
    dst_rgb = (torch.clone(B[:, 3:]) * 0.002).cuda()
    src = torch.ones((4,A.shape[0])).cuda()
    dst = torch.ones((4,B.shape[0])).cuda()
    src[:3,:] = A[:, :3].T.clone()
    dst[:3,:] = B[:, :3].T.clone()

    # apply the initial pose estimation
    if init_pose is not None:
        pose = torch.eye(4).cuda()
        pose[:3, :] = init_pose
        src = torch.mm(pose, src)

    prev_error = 0
    init_distances, _ = my_nearest_neighbor_cuda(torch.cat([dst[:3,:].T, dst_rgb], -1), torch.cat([src[:3,:].T, src_rgb], -1))

    for i in range(max_iterations):
        # find the nearest neighbors between the current source and destination points
        distances, indices = my_nearest_neighbor_cuda(torch.cat([dst[:3,:].T, dst_rgb], -1), torch.cat([src[:3,:].T, src_rgb], -1))
        # print("distance in icp: ", np.mean(distances))
        
        # compute the transformation between the current source and nearest destination points
        T,_,_ = best_fit_transform_torch(src[:3,indices].T, dst[:3,:].T)

        # update the current source
        src = torch.mm(T, src)

        # check error
        mean_error = np.mean(distances)
        if np.abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error

    # calculate final transformation
    # print(mean_error)
    T,_,_ = best_fit_transform_torch(A[:, :3], src[:3,:].T)

    return T, distances, init_distances
def get_all_msk(pcld, all_cld, mask):
    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(pcld)
    distances, indices = neigh.kneighbors(all_cld, return_distance=True)
    all_msk = mask[indices].reshape(-1)

    return all_msk
