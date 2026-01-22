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

import concurrent.futures
import math
import multiprocessing
import os
from argparse import ArgumentParser
from os import makedirs
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import create_render_func
from scene.shell_gaussian_model import ShellGaussianModel
from mesh_renderer import NVDiffRenderer
from scene import CameraDataset, Scene
from scene.dataset_readers import CameraInfo, readCamerasFromTransforms, readMeshesFromTransforms
from utils.camera_utils import cameraList_from_camInfos
from utils.general_utils import safe_state
from utils.sh_utils import RGB2SH

mesh_renderer = NVDiffRenderer()


def write_data(path2data):
    for path, data in path2data.items():
        if not path.parent.exists():
            path.parent.mkdir(parents=True, exist_ok=True)

        if path.suffix in [".png", ".jpg"]:
            data = data.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
            Image.fromarray(data).save(path)
        elif path.suffix in [".obj"]:
            with open(path, "w") as f:
                f.write(data)
        elif path.suffix in [".txt"]:
            with open(path, "w") as f:
                f.write(data)
        elif path.suffix in [".npz"]:
            np.savez(path, **data)
        else:
            raise NotImplementedError(f"Unknown file type: {path.suffix}")

def render_set(dataset : ModelParams, name, iteration, views, gaussians, pipeline, background, render_mesh, skip_gt, use_pts):
    render_func = create_render_func("hybrid_render")
    
    if dataset.select_camera_id != -1:
        name = f"{name}_{dataset.select_camera_id}"
    iter_path = Path(dataset.model_path) / name / f"ours_{iteration}"
    render_path = iter_path / "renders"
    if not skip_gt:
        gts_path = iter_path / "gt"
    if render_mesh:
        render_mesh_path = iter_path / "renders_mesh"

    makedirs(render_path, exist_ok=True)
    if not skip_gt:
        makedirs(gts_path, exist_ok=True)

    views_loader = DataLoader(views, batch_size=None, shuffle=False, num_workers=8)
    max_threads = multiprocessing.cpu_count()
    print("Max threads: ", max_threads)
    worker_args = []
    for idx, view in enumerate(tqdm(views_loader, desc="Rendering progress")):
        gaussians.select_mesh_by_timestep(view.timestep)
        rendering = render_func(view, gaussians, pipeline, background, pts = use_pts)["render"]
        gt = view.original_image[0:3, :, :]
        if render_mesh:
            out_dict = mesh_renderer.render_from_camera(gaussians.verts, gaussians.faces, view)
            rgba_mesh = out_dict["rgba"].squeeze(0).permute(2, 0, 1)  # (C, W, H)
            rgb_mesh = rgba_mesh[:3, :, :]
            alpha_mesh = rgba_mesh[3:, :, :]
            mesh_opacity = 0.5
            rendering_mesh = rgb_mesh * alpha_mesh * mesh_opacity + gt.to(rgb_mesh) * (
                alpha_mesh * (1 - mesh_opacity) + (1 - alpha_mesh)
            )

        path2data = {}
        cam_id = view.image_name[-2:]
        path2data[Path(render_path) / f"{view.timestep:05}_{cam_id}.png"] = rendering
        if not skip_gt:
            path2data[Path(gts_path) / f"{view.timestep:05}_{cam_id}.png"] = gt
        if render_mesh:
            path2data[Path(render_mesh_path) / f"{view.timestep:05}_{cam_id}.png"] = rendering_mesh
        worker_args.append([path2data])

        if len(worker_args) == max_threads or idx == len(views_loader) - 1:
            with concurrent.futures.ThreadPoolExecutor(max_threads) as executor:
                futures = [executor.submit(write_data, *args) for args in worker_args]
                concurrent.futures.wait(futures)
            worker_args = []

    try:
        render_glob = f"{render_path}/*.png"
        render_pngs = list(Path(render_path).glob("*.png"))
        if render_pngs:
            os.system(
                f"ffmpeg -y -framerate 25 -f image2 -pattern_type glob -i '{render_glob}' -pix_fmt yuv420p {iter_path}/renders.mp4"
            )
        else:
            print(f"No rendered PNGs found in {render_path}, skipping renders.mp4")
        if not skip_gt:
            gts_pngs = list(Path(gts_path).glob("*.png"))
            if gts_pngs:
                os.system(
                    f"ffmpeg -y -framerate 25 -f image2 -pattern_type glob -i '{gts_path}/*.png' -pix_fmt yuv420p {iter_path}/gt.mp4"
                )
            else:
                print(f"No GT PNGs found in {gts_path}, skipping gt.mp4")
        if render_mesh:
            mesh_pngs = list(Path(render_mesh_path).glob("*.png"))
            if mesh_pngs:
                os.system(
                    f"ffmpeg -y -framerate 25 -f image2 -pattern_type glob -i '{render_mesh_path}/*.png' -pix_fmt yuv420p {iter_path}/renders_mesh.mp4"
                )
            else:
                print(f"No mesh PNGs found in {render_mesh_path}, skipping renders_mesh.mp4")
    except Exception as e:
        print(e)

def get_forwardpose(n_steps=120):
    z = 1.2
    
    center = np.array([0.0, 0.0, 0.0]).astype(np.float32)
    up = np.array([0.0, 1.0, 0.0], dtype=center.dtype)
    
    theta = math.pi / 12
    diff = np.stack([z * np.sin(theta), 0, z * np.cos(theta)])
    cam_pos = center + diff
    l = -diff / np.linalg.norm(diff)
    s = np.cross(l, up) / np.linalg.norm(np.cross(l, up))
    u = np.cross(s, l) / np.linalg.norm(np.cross(s, l))
    c2w = np.concatenate([np.stack([s, -u, l], axis=1), cam_pos[:, None]], axis=1)
    c2w = np.concatenate([c2w, np.array([[0, 0, 0, 1]], dtype=np.float32)], axis=0)
    
    all_c2w = np.repeat(c2w[None], n_steps, axis=0)
    return all_c2w

def create_spheric_poses(n_steps=120):
    # # normal dist
    z = 1.2

    center = np.array([0.0, 0.02, 0.0]).astype(np.float32)
    r = 0.2
    up = np.array([0.0, 1.0, 0.0], dtype=center.dtype)

    all_c2w = []
    for theta in np.linspace(2 * math.pi, 0, n_steps):
        diff = np.stack([r * np.cos(theta), r * np.sin(theta), z])
        cam_pos = center + diff
        l = -diff / np.linalg.norm(diff)
        s = np.cross(l, up) / np.linalg.norm(np.cross(l, up))
        u = np.cross(s, l) / np.linalg.norm(np.cross(s, l))
        c2w = np.concatenate([np.stack([s, -u, l], axis=1), cam_pos[:, None]], axis=1)
        c2w = np.concatenate([c2w, np.array([[0, 0, 0, 1]], dtype=np.float32)], axis=0)
        all_c2w.append(c2w)

    all_c2w = np.stack(all_c2w, axis=0)

    return all_c2w


def create_zoom_poses(n_steps=120):
    near = 0.2
    far = 3.5

    half_steps = n_steps // 2
    trace_out = np.linspace(near, far, half_steps)
    trace_in = np.linspace(far, near, n_steps - half_steps + 1)
    trace = np.concatenate([trace_out, trace_in[1:]], axis=0)

    center = np.array([0.0, 0.0, 0.0]).astype(np.float32)
    up = np.array([0.0, 1.0, 0.0], dtype=center.dtype)

    all_c2w = []
    for z in trace:
        diff = np.stack([0, 0, z])
        cam_pos = center + diff
        l = -diff / np.linalg.norm(diff)
        s = np.cross(l, up) / np.linalg.norm(np.cross(l, up))
        u = np.cross(s, l) / np.linalg.norm(np.cross(s, l))
        c2w = np.concatenate([np.stack([s, -u, l], axis=1), cam_pos[:, None]], axis=1)
        c2w = np.concatenate([c2w, np.array([[0, 0, 0, 1]], dtype=np.float32)], axis=0)
        all_c2w.append(c2w)

    return all_c2w


def render_json(
    dataset: ModelParams,
    name,
    iteration,
    jsonfile,
    gaussians,
    pipeline,
    background,
    render_mesh,
    free_view,
    num_frames,
    use_pts,
    cam_infos=None,
):
    render_func = create_render_func("hybrid_render")

    iter_path = Path(dataset.model_path) / name / f"ours_{iteration}"
    render_path = iter_path / "renders"
    if render_mesh:
        render_mesh_path = iter_path / "renders_mesh"

    makedirs(render_path, exist_ok=True)

    basedir = os.path.dirname(jsonfile)
    filename = os.path.basename(jsonfile)
    use_ref = True
    if cam_infos is None:
        use_ref = False
        cam_infos = readCamerasFromTransforms(basedir, filename, dataset.white_background)
    num_frames = len(cam_infos) if num_frames == -1 else num_frames
    if free_view:
        poses = create_spheric_poses(num_frames)
        # poses = create_zoom_poses(num_frames)
        # poses = get_forwardpose(num_frames)
        ref_info = cam_infos[0]
        cam_infos = []
        for i in range(num_frames):
            w2c = np.linalg.inv(poses[i])
            R = w2c[:3, :3].T
            t = w2c[:3, 3]

            cam_infos.append(
                CameraInfo(
                    i,
                    R,
                    t,
                    ref_info.FovY,
                    ref_info.FovX,
                    ref_info.bg,
                    ref_info.image_path,
                    ref_info.image_name,
                    ref_info.width,
                    ref_info.height,
                    i,
                    0,
                )
            )

    cameras = cameraList_from_camInfos(cam_infos, 1.0, dataset)
    views = CameraDataset(cameras)

    views_loader = DataLoader(views, batch_size=None, shuffle=False, num_workers=8)
    max_threads = multiprocessing.cpu_count()
    print("Max threads: ", max_threads)
    worker_args = []
    for idx, view in enumerate(tqdm(views_loader, desc="Rendering progress")):
        index = idx if use_ref else view.timestep
        gaussians.select_mesh_by_timestep(index)
        rendering = render_func(view, gaussians, pipeline, background, pts = use_pts)["render"]
        gt = view.original_image[0:3, :, :]
        if render_mesh:
            out_dict = mesh_renderer.render_from_camera(gaussians.verts, gaussians.faces, view)
            rgba_mesh = out_dict["rgba"].squeeze(0).permute(2, 0, 1)  # (C, W, H)
            rgb_mesh = rgba_mesh[:3, :, :]
            alpha_mesh = rgba_mesh[3:, :, :]
            mesh_opacity = 0.5
            rendering_mesh = rgb_mesh * alpha_mesh * mesh_opacity + gt.to(rgb_mesh) * (
                alpha_mesh * (1 - mesh_opacity) + (1 - alpha_mesh)
            )

        path2data = {}
        cam_id = view.image_name[-2:]
        path2data[Path(render_path) / f"{index:05}_{cam_id}.png"] = rendering
        if render_mesh:
            path2data[Path(render_mesh_path) / f"{index:05}_{cam_id}.png"] = rendering_mesh
        worker_args.append([path2data])

        if len(worker_args) == max_threads or idx == len(views_loader) - 1:
            with concurrent.futures.ThreadPoolExecutor(max_threads) as executor:
                futures = [executor.submit(write_data, *args) for args in worker_args]
                concurrent.futures.wait(futures)
            worker_args = []
    
    try:
        os.system(
            f"ffmpeg -y -framerate 25 -f image2 -pattern_type glob -i '{render_path}/*.png' -pix_fmt yuv420p {iter_path}/renders.mp4"
        )
        if render_mesh:
            os.system(
                f"ffmpeg -y -framerate 25 -f image2 -pattern_type glob -i '{render_mesh_path}/*.png' -pix_fmt yuv420p {iter_path}/renders_mesh.mp4"
            )
    except Exception as e:
        print(e)


def render_sets(
    dataset: ModelParams,
    iteration: int,
    pipeline: PipelineParams,
    skip_train: bool,
    skip_val: bool,
    skip_test: bool,
    render_mesh: bool,
    skip_gt: bool,
    use_pts: bool,
    texture_path: str,
    jsonfile=None,
    name=None,
    free_view=False,
    num_frames=-1,
    fix_expr_id=-1,
    cam_info=None
):
    with torch.no_grad():
        gaussians = ShellGaussianModel(dataset.sh_degree)
        
        def load_texture(texture_path):
            if os.path.exists(texture_path):
                image = Image.open(texture_path).convert("RGBA")
                image_array = np.array(image) / 255.0
                rgb = image_array[..., :3]
                alpha = image_array[..., 3]
                
                current_texture = gaussians._textures_dc.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
                if current_texture.shape != rgb.shape:
                    print(f"Error: Texture shape {current_texture.shape} does not match image shape {rgb.shape}")
                    return
                
                mask = alpha > 0
                current_texture[mask] = RGB2SH(rgb[mask])
                
                gaussians.new_textures_dc = nn.Parameter(
                    torch.tensor(current_texture, dtype=torch.float, device="cuda")
                ).permute(2, 0, 1).unsqueeze(0)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        if jsonfile is not None:
            scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, load_cameras=False)
            if texture_path != "":
                load_texture(texture_path)
            
            gaussians.load_all_meshes(
                readMeshesFromTransforms(os.path.dirname(jsonfile), os.path.basename(jsonfile)), num_frames, fix_expr_id
            )
            cam_infos = None
            if cam_info is not None:
                cam_infos = [cam_info for _ in range(gaussians.num_timesteps)]
            render_json(
                dataset,
                f"{name}",
                scene.loaded_iter,
                jsonfile,
                gaussians,
                pipeline,
                background,
                render_mesh,
                free_view,
                num_frames,
                use_pts,
                cam_infos,
            )
            return

        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        if texture_path != "":
            load_texture(texture_path)
        if dataset.target_path != "":
            name = os.path.basename(os.path.normpath(dataset.target_path))
            # when loading from a target path, test cameras are merged into the train cameras
            render_set(
                dataset,
                f"{name}",
                scene.loaded_iter,
                scene.getTrainCameras(),
                gaussians,
                pipeline,
                background,
                render_mesh,
                skip_gt,
                use_pts
            )
        else:
            if not skip_train:
                render_set(
                    dataset,
                    "train",
                    scene.loaded_iter,
                    scene.getTrainCameras(),
                    gaussians,
                    pipeline,
                    background,
                    render_mesh,
                    skip_gt,
                    use_pts
                )

            if not skip_val:
                render_set(
                    dataset,
                    "val",
                    scene.loaded_iter,
                    scene.getValCameras(),
                    gaussians,
                    pipeline,
                    background,
                    render_mesh,
                    skip_gt,
                    use_pts
                )

            if not skip_test:
                render_set(
                    dataset,
                    "test",
                    scene.loaded_iter,
                    scene.getTestCameras(),
                    gaussians,
                    pipeline,
                    background,
                    render_mesh,
                    skip_gt,
                    use_pts
                )


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--jsonfile", type=str, default=None)
    parser.add_argument("--ref_view", type=str, default=None)
    parser.add_argument("--free_view", action="store_true")
    parser.add_argument("--fix_expr_id", type=int, default=-1)
    parser.add_argument("-n", "--num_frames", type=int, default=-1)
    parser.add_argument("--name", type=str, default="free_view")
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_val", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--render_mesh", action="store_true")
    parser.add_argument("--skip_gt", action="store_true")
    parser.add_argument("--use_pts", action="store_true")
    parser.add_argument("--texture_path", default="", type=str)
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    cam_info = None
    # if args.ref_view is not None:
    #     # load view
    #     basedir = os.path.dirname(args.ref_view)
    #     filename = os.path.basename(args.ref_view)
    #     cam_info = readCamerasFromTransforms(basedir, filename, True)[0]

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(
        model.extract(args),
        args.iteration,
        pipeline.extract(args),
        args.skip_train,
        args.skip_val,
        args.skip_test,
        args.render_mesh,
        args.skip_gt,
        args.use_pts,
        args.texture_path,
        # args.jsonfile,
        # args.name,
        # args.free_view,
        # args.num_frames,
        # args.fix_expr_id,
        # cam_info,
    )
