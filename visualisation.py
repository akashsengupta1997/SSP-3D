import torch
import matplotlib.pyplot as plt
from smplx import SMPL

from utils.renderer import Renderer
from utils.cam_utils import perspective_project_torch
from data.ssp3d_dataset import SSP3DDataset
import config

# SMPL models in torch
smpl_male = SMPL(config.SMPL_MODEL_DIR, batch_size=1, gender='male')
smpl_female = SMPL(config.SMPL_MODEL_DIR, batch_size=1, gender='female')

# Pyrender renderer
renderer = Renderer(img_res=512)

# SSP-3D datset class
ssp3d_dataset = SSP3DDataset(config.SSP_3D_PATH)

indices_to_plot = [11, 60, 199]  # Visualising 3 examples from SSP-3D

for i in indices_to_plot:
    data = ssp3d_dataset.__getitem__(i)

    image = data['image']
    cropped_image = data['cropped_image']
    silhouette = data['silhouette']
    joints2D = data['joints2D']

    body_shape = data['shape']
    body_pose = data['pose']
    gender = data['gender']
    cam_trans = data['cam_trans']

    # Obtaining body vertex mesh from SMPL shape and pose
    body_shape = torch.from_numpy(body_shape[None, :]).float()
    body_pose = torch.from_numpy(body_pose[None, :]).float()
    cam_trans = torch.from_numpy(cam_trans[None, :]).float()
    if gender == 'm':
        smpl_output = smpl_male(body_pose=body_pose[:, 3:],
                                global_orient=body_pose[:, :3],  # First 3 axis-angle pose parameters are global body rotation
                                betas=body_shape)
    elif gender == 'f':
        smpl_output = smpl_female(body_pose=body_pose[:, 3:],
                                  global_orient=body_pose[:, :3],  # First 3 axis-angle pose parameters are global body rotation
                                  betas=body_shape)
    vertices = smpl_output.vertices
    projected_vertices = perspective_project_torch(vertices, cam_trans,
                                                   focal_length=config.FOCAL_LENGTH,
                                                   img_wh=512)

    vertices = vertices.cpu().detach().numpy()[0]
    projected_vertices = projected_vertices.cpu().detach().numpy()[0]
    cam_trans = cam_trans.cpu().detach().numpy()[0]

    # Rendering vertex mesh
    rend_img = renderer(vertices, cam_trans, img=image)

    # Visualise
    plt.figure(figsize=(14, 8))
    plt.tight_layout()

    plt.subplot(231)
    plt.gca().set_title('Image')
    plt.gca().set_axis_off()
    plt.imshow(image)

    plt.subplot(232)
    plt.gca().set_title('Cropped Image')
    plt.gca().set_axis_off()
    plt.imshow(cropped_image)

    plt.subplot(233)
    plt.gca().set_title('2D Joints')
    plt.gca().set_axis_off()
    plt.imshow(image)
    for j in range(joints2D.shape[0]):
        plt.scatter(joints2D[j, 0], joints2D[j, 1], s=2, c='r')
        plt.text(joints2D[j, 0], joints2D[j, 1], s=str(j))

    plt.subplot(234)
    plt.gca().set_title('Silhouette')
    plt.gca().set_axis_off()
    plt.imshow(image)
    plt.imshow(silhouette, alpha=0.4)

    plt.subplot(235)
    plt.gca().set_title('Projected Vertices')
    plt.gca().set_axis_off()
    plt.imshow(image)
    plt.scatter(projected_vertices[:, 0], projected_vertices[:, 1], s=1)

    plt.subplot(236)
    plt.gca().set_title('Body Render')
    plt.gca().set_axis_off()
    plt.imshow(rend_img)

    plt.subplots_adjust(top=0.93, bottom=0.07, right=1, left=0, hspace=0.13, wspace=0)
    plt.show()
