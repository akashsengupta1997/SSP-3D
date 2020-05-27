import numpy as np
import pyrender
import trimesh
import math
from pyrender.constants import RenderFlags

import config

class Renderer:
    """
    Renderer used for visualizing the SMPL model
    Code adapted from https://github.com/vchoutas/smplify-x
    """
    def __init__(self, img_res=512):
        self.renderer = pyrender.OffscreenRenderer(viewport_width=img_res,
                                                   viewport_height=img_res,
                                                   point_size=1.0)
        self.camera_center = [img_res // 2, img_res // 2]
        self.faces = np.load(config.SMPL_FACES_PATH)
        self.img_res = img_res

        # set the scene
        self.scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0],
                                    ambient_light=(0.3, 0.3, 0.3))

        light = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=1.)

        light_pose = np.eye(4)
        light_pose[:3, 3] = [0, -1, 1]
        self.scene.add(light, pose=light_pose)

        light_pose[:3, 3] = [0, 1, 1]
        self.scene.add(light, pose=light_pose)

        # light_pose[:3, 3] = [1, 1, 2]
        # self.scene.add(light, pose=light_pose)

    def __call__(self, verts, cam_trans, img=None, angle=None, axis=None, mesh_filename=None,
                 color=[0.8, 0.3, 0.3], return_mask=False):

        mesh = trimesh.Trimesh(verts, self.faces, process=False)
        Rx = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
        mesh.apply_transform(Rx)

        cam_trans[0] *= -1.

        if angle and axis:
            # Apply given mesh rotation to the mesh - useful for rendering from different views
            R = trimesh.transformations.rotation_matrix(math.radians(angle), axis)
            mesh.apply_transform(R)

        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.2,
            alphaMode='OPAQUE',
            baseColorFactor=(color[0], color[1], color[2], 1.0))


        mesh = pyrender.Mesh.from_trimesh(mesh, material=material)
        mesh_node = self.scene.add(mesh, 'mesh')

        camera_pose = np.eye(4)
        camera_pose[:3, 3] = cam_trans
        camera = pyrender.IntrinsicsCamera(fx=config.FOCAL_LENGTH, fy=config.FOCAL_LENGTH,
                                           cx=self.camera_center[0], cy=self.camera_center[1])
        cam_node = self.scene.add(camera, pose=camera_pose)

        rgb, rend_depth = self.renderer.render(self.scene, flags=RenderFlags.RGBA)
        valid_mask = (rend_depth > 0)
        if return_mask:
            return valid_mask
        else:
            if img is None:
                img = np.zeros((self.img_res, self.img_res, 3))
            valid_mask = valid_mask[:, :, None]
            output_img = rgb[:, :, :-1] * valid_mask + (1 - valid_mask) * img
            image = output_img.astype(np.uint8)

            self.scene.remove_node(mesh_node)
            self.scene.remove_node(cam_node)

            return image


