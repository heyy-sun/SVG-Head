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
import torch
from torch.utils.data import DataLoader
from utils.loss_utils import l1_loss, ssim, gradient_smooth, alpha_loss
from gaussian_renderer import create_render_func, network_gui
from mesh_renderer import NVDiffRenderer
import sys
from scene import Scene, GaussianModel, ShellGaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr, error_map
from lpipsPyTorch import lpips
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import torch.nn.functional as F
from pytorch3d.loss import mesh_laplacian_smoothing, mesh_normal_consistency
from pytorch3d.structures import Meshes
from utils.log_utils import save_side_by_side, generate_image_5, save_uv_map
from PIL import Image
import numpy as np

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint, debug_from):
    first_iter = 0
    generated_images_5 = {"996":None, "997":None, "998":None, "999":None, "0":None}
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = ShellGaussianModel(dataset.sh_degree, dataset.disable_flame_static_offset, dataset.not_finetune_flame_params, dataset.not_finetune_xyz)
    mesh_renderer = NVDiffRenderer()
    scene = Scene(dataset, gaussians, init_from_iteration=dataset.init_from_iteration)
    gaussians.training_setup(opt)

    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)
    
    loader_camera_train = DataLoader(scene.getTrainCameras(), batch_size=None, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True)
    iter_camera_train = iter(loader_camera_train)
    # viewpoint_stack = None
    ema_loss_for_log = 0.0
    if dataset.init_from_iteration != None:
        first_iter = dataset.init_from_iteration
        gaussians.set_stage(0 if first_iter < opt.add_gaussians_iteration else 1, opt)
        if first_iter >= opt.add_gaussians_iteration:
            image_no_expr_sh = None
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    render_func = create_render_func("hybrid_render")
    
    for iteration in range(first_iter, opt.iterations + 1):

        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                # receive data
                net_image = None
                # custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer, use_original_mesh = network_gui.receive()
                custom_cam, msg = network_gui.receive()

                # render
                if custom_cam != None:
                    # mesh selection by timestep
                    gaussians.select_mesh_by_timestep(custom_cam.timestep, msg['use_original_mesh'])
                    
                    # gaussian splatting rendering
                    if msg['show_splatting']:
                        net_image = render_func(custom_cam, gaussians, pipe, background, msg['scaling_modifier'])["render"]
                    
                    # mesh rendering
                    if msg['show_mesh']:
                        out_dict = mesh_renderer.render_from_camera(gaussians.verts, gaussians.faces, custom_cam)

                        rgba_mesh = out_dict['rgba'].squeeze(0).permute(2, 0, 1)  # (C, W, H)
                        rgb_mesh = rgba_mesh[:3, :, :]
                        alpha_mesh = rgba_mesh[3:, :, :]

                        mesh_opacity = msg['mesh_opacity']
                        if net_image is None:
                            net_image = rgb_mesh
                        else:
                            net_image = rgb_mesh * alpha_mesh * mesh_opacity  + net_image * (alpha_mesh * (1 - mesh_opacity) + (1 - alpha_mesh))

                    # send data
                    net_dict = {'num_timesteps': gaussians.num_timesteps, 'num_points': gaussians.get_xyz.shape[0]}
                    network_gui.send(net_image, net_dict)
                if msg['do_training'] and ((iteration < int(opt.iterations)) or not msg['keep_alive']):
                    break
            except Exception as e:
                # print(e)
                network_gui.conn = None

        iter_start.record()

        # optimizing the learning rate of bc
        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        try:
            viewpoint_cam = next(iter_camera_train)
        except StopIteration:
            iter_camera_train = iter(loader_camera_train)
            viewpoint_cam = next(iter_camera_train)

        gaussians.select_mesh_by_timestep(viewpoint_cam.timestep)

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        # bg = torch.rand((3), device="cuda") if opt.random_background else background
        render_pkg = render_func(viewpoint_cam, gaussians, pipe, background, pts = gaussians.current_num_shells == 2)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        render_pkg_no_expr_sh = render_func(viewpoint_cam, gaussians, pipe, background, expr = False, sh = False, pts = False)
        if (gaussians.current_num_shells == 1):
            image_no_expr_sh = render_pkg_no_expr_sh["render"]
        else:
            alpha_no_expr_sh = render_pkg_no_expr_sh["alpha"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()

        losses = {}
        losses['l1'] = l1_loss(image, gt_image) * (1.0 - opt.lambda_dssim)
        losses['ssim'] = (1.0 - ssim(image, gt_image)) * opt.lambda_dssim

        if (opt.lambda_no_expr_sh != 0 and gaussians.current_num_shells == 1):
            losses['l1_no_expr_sh'] = l1_loss(image_no_expr_sh, gt_image) * (1.0 - opt.lambda_dssim) * opt.lambda_no_expr_sh
            losses['ssim_no_expr_sh'] = (1.0 - ssim(image_no_expr_sh, gt_image)) * opt.lambda_dssim * opt.lambda_no_expr_sh
        
        if opt.lambda_xyz != 0 and gaussians.current_num_shells == 2:
            start_idx = gaussians.calculate_idx()
            visibility_filter_layer = visibility_filter[start_idx[1]:start_idx[2]]
            if opt.metric_xyz:
                losses['xyz'] = F.relu((gaussians._xyz*gaussians.face_scaling[gaussians.face_num[1]])[visibility_filter_layer] - opt.threshold_xyz).norm(dim=1).mean() * opt.lambda_xyz
            else:
                losses['xyz'] = F.relu(gaussians._xyz[visibility_filter_layer].norm(dim=1) - opt.threshold_xyz).mean() * opt.lambda_xyz
        
        if opt.lambda_scale != 0:
            if opt.metric_scale:
                losses['scale'] = F.relu(gaussians.get_scaling[visibility_filter] - opt.threshold_scale).norm(dim=1).mean() * opt.lambda_scale
            else:
                losses['scale'] = F.relu(torch.exp(gaussians.get_combined_scaling[visibility_filter]) - opt.threshold_scale).norm(dim=1).mean() * opt.lambda_scale
        
        if opt.lambda_alpha != 0 and gaussians.current_num_shells == 2:
            losses['alpha'] = alpha_loss(alpha_no_expr_sh) * opt.lambda_alpha

        losses['total'] = sum([v for k, v in losses.items()])

        losses['total'].backward()

        for param_group in gaussians.optimizer.param_groups:
            for param in param_group['params']:
                if param.grad is not None:
                    if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                        print(f"{iteration} {param_group['name']} Gradient contains NaN or Inf!")

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * losses['total'].item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                postfix = {"Loss": f"{ema_loss_for_log:.{7}f}"}
                if 'scale' in losses:
                    postfix["scale"] = f"{losses['scale']:.{7}f}"
                if 'laplacian' in losses:
                    postfix["laplacian"] = f"{losses['laplacian']:.{7}f}"
                if 'normal' in losses:
                    postfix["normal"] = f"{losses['normal']:.{7}f}"
                progress_bar.set_postfix(postfix)
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, losses, iter_start.elapsed_time(iter_end), testing_iterations, scene, render_func, (pipe, background))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            if opt.add_densification:
                # Densification
                if iteration < opt.densify_until_iter:
                    # Keep track of max radii in image-space for pruning
                    start_idx = gaussians.calculate_idx()
                    current_layer = gaussians.current_num_shells - 1
                    visibility_filter_layer = visibility_filter[start_idx[current_layer]:start_idx[current_layer+1]]
                    if visibility_filter_layer.sum() != 0:
                        gaussians.max_radii2D[current_layer][visibility_filter_layer] = torch.max(gaussians.max_radii2D[current_layer][visibility_filter_layer], radii[start_idx[current_layer]:start_idx[current_layer+1]][visibility_filter_layer])
                    gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                    if ((iteration > opt.densify_from_iter and iteration <= opt.add_gaussians_iteration) or iteration > opt.densify_from_iter + opt.add_gaussians_iteration) and iteration % opt.densification_interval == 0:
                        size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                        gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, current_layer)
                    
                    if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                        gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            gaussians.select_mesh_by_timestep(viewpoint_cam.timestep)

            output_dir = os.path.join(dataset.model_path, "image")
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            # Save obj file
            if iteration == 1 or iteration % 5000 == 0:
                gaussians.write_obj(filepath = os.path.join(output_dir, str(iteration)+".obj"), verts = gaussians.get_xyz.detach().cpu().numpy(), log = False)
                gaussians.write_obj(filepath = os.path.join(output_dir, str(iteration)+"_FLAME.obj"), verts = gaussians.verts.squeeze(0).detach().cpu().numpy(), tris = gaussians.faces.detach().cpu().numpy(), log = False)

            # Save offset & texture maps
            if (iteration == 1 or iteration % 1000 == 0):
                save_uv_map(gaussians.get_textures_dc, os.path.join(output_dir, str(iteration) + "_t_dc"), rgb = True)
                save_uv_map(gaussians.get_textures_expr, os.path.join(output_dir, str(iteration) + "_t_expr"), rgb = True)    
            
            # Output image
            if iteration == 1:
                save_side_by_side([generate_image_5(gaussians, viewpoint_cam, mesh_renderer, image, image_no_expr_sh, gt_image)], output_dir, iteration, vertical = True)
            elif str(iteration % 1000) in generated_images_5:
                generated_images_5[str(iteration % 1000)] = generate_image_5(gaussians, viewpoint_cam, mesh_renderer, image, image_no_expr_sh, gt_image)
            if generated_images_5["0"] != None:
                if generated_images_5["996"] != None:
                    save_side_by_side([generated_images_5["996"], generated_images_5["997"], generated_images_5["998"], generated_images_5["999"], generated_images_5["0"]], output_dir, iteration, vertical = True)
                generated_images_5 = {"996":None, "997":None, "998":None, "999":None, "0":None}
            
            # Add layer
            if iteration == opt.add_gaussians_iteration:
                gaussians.add_gaussians(opt)
                image_no_expr_sh = None

def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, losses, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', losses['l1'].item(), iteration)
        tb_writer.add_scalar('train_loss_patches/ssim_loss', losses['ssim'].item(), iteration)
        if 'scale' in losses:
            tb_writer.add_scalar('train_loss_patches/scale_loss', losses['scale'].item(), iteration)
        if 'laplacian' in losses:
            tb_writer.add_scalar('train_loss_patches/laplacian_loss', losses['laplacian'].item(), iteration)
        if 'normal' in losses:
            tb_writer.add_scalar('train_loss_patches/normal_loss', losses['normal'].item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', losses['total'].item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        print("[ITER {}] Evaluating".format(iteration))
        torch.cuda.empty_cache()
        validation_configs = (
            {'name': 'val', 'cameras' : scene.getValCameras()},
            {'name': 'test', 'cameras' : scene.getTestCameras()},
        )

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                ssim_test = 0.0
                lpips_test = 0.0
                num_vis_img = 10
                image_cache = []
                gt_image_cache = []
                vis_ct = 0
                for idx, viewpoint in tqdm(enumerate(DataLoader(config['cameras'], shuffle=False, batch_size=None, num_workers=8)), total=len(config['cameras'])):
                    if hasattr(scene.gaussians, "select_mesh_by_timestep"):
                        scene.gaussians.select_mesh_by_timestep(viewpoint.timestep)
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and ((len(config['cameras']) < num_vis_img) or (idx % (len(config['cameras']) // num_vis_img) == 0)):
                        tb_writer.add_images(config['name'] + "_{}/render".format(vis_ct), image[None], global_step=iteration)
                        error_image = error_map(image, gt_image)
                        tb_writer.add_images(config['name'] + "_{}/error".format(vis_ct), error_image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_{}/ground_truth".format(vis_ct), gt_image[None], global_step=iteration)
                        vis_ct += 1
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    ssim_test += ssim(image, gt_image).mean().double()

                    image_cache.append(image)
                    gt_image_cache.append(gt_image)

                    if idx == len(config['cameras']) - 1 or len(image_cache) == 16:
                        batch_img = torch.stack(image_cache, dim=0)
                        batch_gt_img = torch.stack(gt_image_cache, dim=0)
                        lpips_test += lpips(batch_img, batch_gt_img).sum().double()
                        image_cache = []
                        gt_image_cache = []

                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                lpips_test /= len(config['cameras'])          
                ssim_test /= len(config['cameras'])          
                print("[ITER {}] Evaluating {}: L1 {:.4f} PSNR {:.4f} SSIM {:.4f} LPIPS {:.4f}".format(iteration, config['name'], l1_test, psnr_test, ssim_test, lpips_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - ssim', ssim_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - lpips', lpips_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--interval", type=int, default=60_000, help="A shared iteration interval for test and saving results and checkpoints.")
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    if len(args.test_iterations) == 0:
        args.test_iterations.extend(list(range(args.interval, args.iterations, args.interval)) + [args.iterations])
    if len(args.save_iterations) == 0:
        args.save_iterations.extend(list(range(args.interval, args.iterations, args.interval)) + [args.iterations])
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
