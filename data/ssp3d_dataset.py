import cv2
import numpy as np
import os
import torch
from torch.utils.data import Dataset

from utils.image_utils import convert_bbox_centre_hw_to_corners


class SSP3DDataset(Dataset):
    def __init__(self, ssp3d_dir_path):
        super(SSP3DDataset, self).__init__()

        # Paths
        self.images_dir = os.path.join(ssp3d_dir_path, 'images')
        self.silhouettes_dir = os.path.join(ssp3d_dir_path, 'silhouettes')

        # Data
        data = np.load(os.path.join(ssp3d_dir_path, 'labels.npz'))

        self.image_fnames = data['fnames']
        self.body_shapes = data['shapes']
        self.body_poses = data['poses']
        self.cam_trans = data['cam_trans']
        self.joints2D = data['joints2D']
        self.bbox_centres = data['bbox_centres']  # Tight bounding box centre
        self.bbox_whs = data['bbox_whs']  # Tight bounding box width/height
        self.genders = data['genders']

    def __len__(self):
        return len(self.image_fnames)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        fname = self.image_fnames[index]
        image = cv2.imread(os.path.join(self.images_dir, fname))[:,:,::-1]
        silhouette = cv2.imread(os.path.join(self.silhouettes_dir, fname), 0)
        joints2D = self.joints2D[index]
        shape = self.body_shapes[index]
        pose = self.body_poses[index]
        cam_trans = self.cam_trans[index]
        gender = self.genders[index]

        # Crop images to bounding box if needed
        bbox_centre = self.bbox_centres[index]
        bbox_wh = self.bbox_whs[index]
        bbox_corners = convert_bbox_centre_hw_to_corners(bbox_centre, bbox_wh, bbox_wh)
        bbox_corners[bbox_corners < 0] = 0
        top_left = bbox_corners[:2].astype(np.int16)
        bottom_right = bbox_corners[2:].astype(np.int16)
        top_left[top_left < 0] = 0
        cropped_image = image[top_left[0]: bottom_right[0], top_left[1]: bottom_right[1]]

        return {'image': image,
                'cropped_image': cropped_image,
                'silhouette': silhouette,
                'joints2D': joints2D,
                'shape': shape,
                'pose': pose,
                'cam_trans': cam_trans,
                'fname': fname,
                'gender': gender}
