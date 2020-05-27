# SSP-3D
Repository for the Sports Shape and Pose 3D (SSP-3D) dataset, as introduced in "Synthetic Training for Accurate 3D Human
003 Pose and Shape Estimation in the Wild". 

![teaser](teaser.png)

## Description
SSP-3D is an evaluation dataset consisting of 311 images of sportspersons in tight-fitted clothes, with a variety of body shapes and poses. Pseudo-ground-truth 3D pose and shape labels (using the SMPL body model) were obtained via multi-frame optimisation with shape consistency between frames (as described in the paper above). The image above shows a few samples from the SSP-3D dataset (under 'Optimised Pose and Shape') as well as a comparison with pre-optimised body predictions (using [VIBE](https://github.com/mkocabas/VIBE)), which demonstrates the improvement in body model fit achieved using optimisation.

## Data
Since SSP-3D is a small dataset, the zip file containing all the necessary data is a part of this repository. Unzipping it will reveal a folder with images, a folder with silhouette annotations and a file called `labels.npz`. This file contains arrays with filenames, SMPL pose parameters, SMPL shape parameters, genders, 2D joint annotations, camera translations and bounding boxes for each image. 
