#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
from copy import deepcopy
import random
import json
from typing import Union, List
import numpy as np
import torch
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.cameras import Camera
from scene.gaussian_model import GaussianModel
from scene.shell_gaussian_model import ShellGaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
from utils.general_utils import PILtoTorch
from PIL import Image

class CameraDataset(torch.utils.data.Dataset):
    def __init__(self, cameras: List[Camera]):
        self.cameras = cameras

    def __len__(self):
        return len(self.cameras)

    def __getitem__(self, idx):
        if isinstance(idx, int):
            # ---- from readCamerasFromTransforms() ----
            camera = deepcopy(self.cameras[idx])

            image = Image.open(camera.image_path)
            im_data = np.array(image.convert("RGBA"))
            '''parsing_path = camera.image_path.replace("images", "parsing").replace(".png", "_labels.png")
            parsing = Image.open(parsing_path)
            parsing_data = np.array(parsing)
            shoulder_bg_mask = (parsing_data == 18) | (parsing_data == 0)
            im_data = np.array(image.convert("RGBA"))
            im_data[shoulder_bg_mask] = (np.array([camera.bg[0], camera.bg[1], camera.bg[2], 1]) * 255).astype(np.uint8)'''

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + camera.bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray((arr * 255.0).astype(np.uint8), "RGB")

            # ---- from loadCam() and Camera.__init__() ----
            resized_image_rgb = PILtoTorch(image, (camera.image_width, camera.image_height))

            image = resized_image_rgb[:3, ...]

            if resized_image_rgb.shape[1] == 4:
                gt_alpha_mask = resized_image_rgb[3:4, ...]
                image *= gt_alpha_mask
            
            camera.original_image = image.clamp(0.0, 1.0)
            return camera
        elif isinstance(idx, slice):
            return CameraDataset(self.cameras[idx])
        else:
            raise TypeError("Invalid argument type")

class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : Union[GaussianModel, ShellGaussianModel], load_iteration=None, init_from_iteration=None, shuffle=True, resolution_scales=[1.0], load_cameras=True):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "checkpoint"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        # load dataset
        if load_cameras:
            assert os.path.exists(args.source_path), "Source path does not exist: {}".format(args.source_path)
            if os.path.exists(os.path.join(args.source_path, "sparse")):
                scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval)
            elif os.path.exists(os.path.join(args.source_path, "canonical_flame_param.npz")):
                print("Found FLAME parameter, assuming dynamic NeRF data set!")
                scene_info = sceneLoadTypeCallbacks["DynamicNerf"](args.source_path, args.white_background, args.eval, target_path=args.target_path)
            elif os.path.exists(os.path.join(args.source_path, "transforms.json")):
                print("Found transforms_train.json file, assuming Blender data set!")
                scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
            else:
                assert False, "Could not recognize scene type!"

            # process cameras
            self.train_cameras = {}
            self.val_cameras = {}
            self.test_cameras = {}

            if not self.loaded_iter:
                if gaussians.face_num == None:
                    with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                        dest_file.write(src_file.read())
                json_cams = []
                camlist = []
                if scene_info.test_cameras:
                    camlist.extend(scene_info.test_cameras)
                if scene_info.train_cameras:
                    camlist.extend(scene_info.train_cameras)
                if scene_info.val_cameras:
                    camlist.extend(scene_info.val_cameras)
                for id, cam in enumerate(camlist):
                    json_cams.append(camera_to_JSON(id, cam))
                with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                    json.dump(json_cams, file)

            if shuffle:
                random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
                random.shuffle(scene_info.val_cameras)  # Multi-res consistent random shuffling
                random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

            self.cameras_extent = scene_info.nerf_normalization["radius"]

            for resolution_scale in resolution_scales:
                print("Loading Training Cameras")
                self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
                print("Loading Validation Cameras")
                self.val_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.val_cameras, resolution_scale, args)
                print("Loading Test Cameras")
                self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)

            # process meshes
            self.gaussians.load_meshes(scene_info.train_meshes, scene_info.test_meshes, 
                                        scene_info.tgt_train_meshes, scene_info.tgt_test_meshes)
        
        # create gaussians
        if self.loaded_iter:
            self.gaussians.load_params(
                os.path.join(self.model_path,
                            "checkpoint",
                            "iteration_" + str(self.loaded_iter),
                            "params.pth"),
                has_target=args.target_path != "",
            )
        else:
            if gaussians.face_num == None:
                self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)
            else:
                self.gaussians.create_from_mesh(self.cameras_extent)
            if init_from_iteration != None:
                self.gaussians.load_params(
                    os.path.join(args.init_from_path,
                                "checkpoint",
                                "iteration_" + str(init_from_iteration),
                                "params.pth"),
                    has_target=args.target_path != "",
                )

    def save(self, iteration):
        checkpoint_path = os.path.join(self.model_path, "checkpoint/iteration_{}".format(iteration))
        self.gaussians.save_params(os.path.join(checkpoint_path, "params.pth"))
        # torch.save((self.gaussians.capture(), iteration),os.path.join(checkpoint_path, str(iteration) + ".pth"))

    def getTrainCameras(self, scale=1.0):
        return CameraDataset(self.train_cameras[scale])
    
    def getValCameras(self, scale=1.0):
        return CameraDataset(self.val_cameras[scale])

    def getTestCameras(self, scale=1.0):
        return CameraDataset(self.test_cameras[scale])