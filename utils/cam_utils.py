import torch
import numpy as np


def perspective_project_torch(points, translation, rotation=None, cam_K=None,
                              focal_length=None, img_wh=None):
    """
    This function computes the perspective projection of a set of points in torch.
    Input:
        points (bs, N, 3): 3D points
        translation (bs, 3): Camera translation
        rotation (bs, 3, 3): Camera rotation
        Either
        cam_K (bs, 3, 3): Camera intrinsics matrix
        Or
        focal_length (bs,) or scalar: Focal length
        camera_center (bs, 2): Camera center
    """
    batch_size = points.shape[0]
    if cam_K is None:
        cam_K = torch.from_numpy(get_intrinsics_matrix(img_wh, img_wh, focal_length).astype(
            np.float32))
        cam_K = torch.cat(batch_size * [cam_K[None, :, :]], dim=0)
        cam_K = cam_K.to(points.device)

    if rotation is None:
        rotation = torch.eye(3).to(points.device)
        rotation = rotation[None, :, :].expand(batch_size, -1, -1)

    # Transform points
    points = torch.einsum('bij,bkj->bki', rotation, points)
    points = points + translation.unsqueeze(1)

    # Apply perspective distortion
    projected_points = points / points[:, :, -1].unsqueeze(-1)

    # Apply camera intrinsics
    projected_points = torch.einsum('bij,bkj->bki', cam_K, projected_points)

    return projected_points[:, :, :-1]


def get_intrinsics_matrix(img_width, img_height, focal_length):
    """
    Camera intrinsic matrix (calibration matrix) given focal length and img_width and
    img_height. Assumes that principal point is at (width/2, height/2).
    """
    K = np.array([[focal_length, 0., img_width/2.0],
                  [0., focal_length, img_height/2.0],
                  [0., 0., 1.]])
    return K