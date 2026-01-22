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

import torch
from torch.utils.data import DataLoader
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
import multiprocessing
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import numpy as np

from gaussian_renderer import create_render_func
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from scene.shell_gaussian_model import ShellGaussianModel
from utils.loss_utils import l1_loss, ssim
from utils.image_utils import psnr
from lpipsPyTorch import lpips

def eval_set(dataset : ModelParams, name, iteration, views, gaussians, pipeline, background, use_pts: bool):
    render_func = create_render_func("hybrid_render")
    if dataset.select_camera_id != -1:
        name = f"{name}_{dataset.select_camera_id}"
    iter_path = Path(dataset.model_path) / name / f"ours_{iteration}"
    render_path = iter_path / "renders"
    gts_path = iter_path / "gt"

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    views_loader = DataLoader(views, batch_size=None, shuffle=False, num_workers=8)
    max_threads = multiprocessing.cpu_count()
    print('Max threads: ', max_threads)
    l1_test = 0.0
    psnr_test = 0.0
    ssim_test = 0.0
    lpips_test = 0.0
    image_cache = []
    gt_image_cache = []

    for idx, view in enumerate(tqdm(views_loader, desc="Evaluating progress")):
        gaussians.select_mesh_by_timestep(view.timestep)
        rendering = torch.clamp(render_func(view, gaussians, pipeline, background, pts=use_pts)["render"], 0.0, 1.0).cuda()
        gt = torch.clamp(view.original_image[0:3, :, :], 0.0, 1.0).cuda()

        l1_test += l1_loss(rendering, gt).mean().double()
        psnr_test += psnr(rendering, gt).mean().double()
        ssim_test += ssim(rendering, gt).mean().double()

        image_cache.append(rendering)
        gt_image_cache.append(gt)

        if idx == len(views) - 1 or len(image_cache) == 16:
            batch_img = torch.stack(image_cache, dim=0)
            batch_gt_img = torch.stack(gt_image_cache, dim=0)
            lpips_test += lpips(batch_img, batch_gt_img).sum().double()
            image_cache = []
            gt_image_cache = []

    psnr_test /= len(views)
    l1_test /= len(views)          
    lpips_test /= len(views)          
    ssim_test /= len(views)          
    print("[ITER {}] Evaluating {}: L1 {:.4f} PSNR {:.4f} SSIM {:.4f} LPIPS {:.4f}".format(iteration, name, l1_test, psnr_test, ssim_test, lpips_test))

def eval_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, use_pts: bool):
    with torch.no_grad():
        gaussians = ShellGaussianModel(dataset.sh_degree, dataset.disable_flame_static_offset, dataset.not_finetune_flame_params, dataset.not_finetune_xyz)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if dataset.target_path != "":
            name = os.path.basename(os.path.normpath(dataset.target_path))
            # when loading from a target path, test cameras are merged into the train cameras
            eval_set(dataset, f'{name}', scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, use_pts)
        else:
            eval_set(dataset, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, use_pts)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--use_pts", action="store_true")
    args = get_combined_args(parser)
    print("Evaluating " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    eval_sets(model.extract(args), args.iteration, pipeline.extract(args), args.use_pts)