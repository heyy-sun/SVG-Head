import torch
import math
from diff_gauss_hybrid_tex import GaussianRasterizationSettings, GaussianRasterizer
from scene.shell_gaussian_model import ShellGaussianModel
from utils.sh_utils import SH2RGB
from PIL import Image
import numpy as np

def save_image(image, path):
    """
    Save image to disk.
    """
    # 转换形状为 (height, width, 3) 以符合 RGB 图像格式
    img_array = image.cpu().detach().numpy()
    img_array = np.clip(SH2RGB(img_array) * 255, 0, 255).astype(np.uint8)

    # 将 NumPy 数组转换为 Pillow 图像
    img_array = Image.fromarray(img_array)
    img_array.save(path)

def tex_render(viewpoint_camera, pc : ShellGaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, expr = True, extra_attrs = None, sh = True):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_bc, dtype=pc.get_bc.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    
    start_idx = pc.calculate_idx()

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform.cuda(),
        projmatrix=viewpoint_camera.full_proj_transform.cuda(),
        sh_degree=pc.active_sh_degree if sh else 0,
        campos=viewpoint_camera.camera_center.cuda(),
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz[start_idx[0]:start_idx[1]]
    means2D = screenspace_points[start_idx[0]:start_idx[1]]
    opacity = pc.get_opacity[start_idx[0]:start_idx[1]]

    scales = pc.get_scaling[start_idx[0]:start_idx[1]]
    rotations = pc.get_rotation[start_idx[0]:start_idx[1]]

    shs = pc.get_features_rest[start_idx[0]:start_idx[1]]
    texture = pc.get_textures(expr)
    uvs = pc.get_uvs
    grad_uvs = pc.get_grad_uvs

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, rendered_depth, rendered_norm, rendered_alpha, radii, extra = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        uvs = uvs,
        gradient_uvs = grad_uvs,
        texture = texture,
        extra_attrs = extra_attrs)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "depth": rendered_depth,
            "norm": rendered_norm,
            "alpha": rendered_alpha,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "extra": extra,
            "radii": radii}