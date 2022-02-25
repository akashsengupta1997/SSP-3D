import torch
import matplotlib.pyplot as plt
import pickle
import cv2
from smplx import SMPL

from utils.renderer import Renderer
from utils.cam_utils import perspective_project_torch
from data.ssp3d_dataset import SSP3DDataset
import config


# SMPL models in torch
smpl_male = SMPL(config.SMPL_MODEL_DIR, batch_size=1, gender='male')
smpl_female = SMPL(config.SMPL_MODEL_DIR, batch_size=1, gender='female')

# Pyrender renderer
renderer = Renderer(faces=smpl_male.faces, img_res=512)

# SSP-3D datset class
ssp3d_dataset = SSP3DDataset(config.SSP_3D_PATH)

f = open('IO/input/mousavi.pkl', 'rb')
data = pickle.load(f)

data['image'] = cv2.cvtColor(cv2.imread("IO/input/mousavi.png", 1), cv2.COLOR_BGR2RGB)
data['silhouette'] = cv2.imread("IO/input/mousavi_mask.png", 0)

image = data['image']
silhouette = data['silhouette']

global_orient = data['global_orient']
betas = data['betas']
body_pose = data['body_pose']
gender = 'm'
cam_trans = data['camera_translation']

# Obtaining body vertex mesh from SMPL shape and pose
betas = torch.from_numpy(betas[None, :]).float()[0]
body_pose = torch.from_numpy(body_pose[None, :]).float()[0]
global_orient = torch.from_numpy(global_orient[None, :]).float()[0]
cam_trans = torch.from_numpy(cam_trans[None, :]).float()[0]

if gender == 'm': 
    smpl_output = smpl_male(body_pose=body_pose,
                            global_orient=global_orient,  
                            betas=betas)
elif gender == 'f':
    smpl_output = smpl_female(body_pose=body_pose,
                                global_orient=global_orient, 
                                betas=betas)
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
fig = plt.figure(figsize=(14, 8))
# fig.suptitle(fname)
plt.tight_layout()

plt.subplot(231)
plt.gca().set_title('Image')
plt.gca().set_axis_off()
plt.imshow(image)

plt.subplot(232)
plt.gca().set_title('Cropped Image')
plt.gca().set_axis_off()

plt.subplot(233)
plt.gca().set_title('2D Joints')
plt.gca().set_axis_off()
plt.imshow(image)

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

plt.subplots_adjust(top=0.93, bottom=0.0, right=1, left=0, hspace=0.13, wspace=0)
plt.show()
