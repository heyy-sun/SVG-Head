import os
from PIL import Image
import numpy as np
import torch
import matplotlib.pyplot as plt
from utils.sh_utils import SH2RGB
from matplotlib import cm
from matplotlib.colors import Normalize

def generate_image_5(gaussians, viewpoint_cam, mesh_renderer, image, image_no_expr_sh, gt_image):
    original_render = mesh_renderer.render_from_camera(gaussians.verts, gaussians.faces, viewpoint_cam)
    original_render_image = tensor_to_pil(original_render['rgba'])

    image_pil = get_image_pil(image)
    gt_image_pil = get_image_pil(gt_image)
    image_no_expr_sh_pil = get_image_pil(image_no_expr_sh) if image_no_expr_sh is not None else None
    
    blended_original = blend_images(image_pil, original_render_image, alpha=0.5)

    # 合并五张图片
    images = [gt_image_pil, image_pil, image_no_expr_sh_pil, blended_original, original_render_image]
    return save_side_by_side(images)

def get_image_pil(image):
    _image = image.permute(1, 2, 0).contiguous().cpu().numpy()
    return Image.fromarray((np.clip(_image, 0, 1) * 255).astype(np.uint8))

def save_side_by_side(images, output_dir = None, iteration = None, vertical = False):
    # 获取每张图片的尺寸，假设所有图片尺寸相同
    width, height = images[0].size
    images = [item for item in images if item is not None]

    if not vertical:
        # 创建一个新的空白图片，宽度是所有图片的总和，高度是最高的那张
        combined_width = width * len(images)
        combined_height = height
        combined_image = Image.new('RGB', (combined_width, combined_height))

        # 将所有图片粘贴到新图片中
        for i, img in enumerate(images):
            combined_image.paste(img, (i * width, 0))
    else:
        # 创建一个新的空白图片，高度是所有图片的总和，宽度是最高的那张
        combined_width = width
        combined_height = height * len(images)
        combined_image = Image.new('RGB', (combined_width, combined_height))

        # 将所有图片粘贴到新图片中
        for i, img in enumerate(images):
            combined_image.paste(img, (0, i * height))

    if iteration != None:
        # 保存合成的图片
        output_path = os.path.join(output_dir, str(iteration) + '_combined.png')
        combined_image.save(output_path)

    return combined_image

def tensor_to_pil(tensor):
    # 将张量从GPU移动到CPU，并转换为numpy数组
    image = tensor.detach().cpu().numpy()
    # 去除batch维度，假设batch size为1
    image = image.squeeze(0)
    # 转换为uint8格式
    image = (image * 255).astype(np.uint8)
    # 创建PIL图片
    return Image.fromarray(image)

def blend_images(base_image, overlay_image, alpha=0.5):
    return Image.blend(base_image.convert("RGB"), overlay_image.convert("RGB"), alpha)

def save_image_with_text(img_tensor, img_path):
    # 去掉第一个维度，将 Tensor 转换为 2D 数组
    img_array = img_tensor.squeeze().cpu().detach().numpy()

    # 使用 matplotlib 显示并保存图像
    plt.imshow(img_array, cmap='viridis')  # 'viridis' 是一个常用的颜色映射
    plt.colorbar()  # 添加颜色条以显示数值范围

    # 保存为 PNG 文件
    plt.savefig(img_path, dpi=300)
    plt.close()

def save_uv_map(img_list, img_path, rgb = False, clip = True):
    if rgb == False:
        # 获取 'viridis' 颜色映射并设置归一化范围
        vmin, vmax = img_list.min().item(), img_list.max().item()  # 根据数据范围设置最小值和最大值
        norm = Normalize(vmin=vmin, vmax=vmax)
        cmap = cm.ScalarMappable(norm=norm, cmap='viridis')

        img_list = img_list.squeeze(0)
        # img_list 的形状是 (1, num_shells - 1, height, width)
        img_list = img_list.squeeze(0)
        for idx in range(img_list.shape[0]):
            img_array = img_list[idx].cpu().detach().numpy()

            img_array = cmap.to_rgba(img_array)  # 映射为 (height, width, 4) 的 RGBA 图像
            img_array = np.clip(img_array[:, :, :3] * 255, 0, 255).astype(np.uint8)  # 转换为 RGB

            # 将 NumPy 数组转换为 Pillow 图像
            img_array = Image.fromarray(img_array)
            img_path_group = f"{img_path}_{idx}.png"
            img_array.save(img_path_group)
    else:
        # img_list 里每个元素的形状是 (1, channels, height, width)
        if isinstance(img_list, torch.Tensor):
            img_list = [img_list]
        
        for idx, img in enumerate(img_list):
            # 转换形状为 (height, width, 3) 以符合 RGB 图像格式
            img_array = img.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
            if clip:
                img_array = np.clip(SH2RGB(img_array) * 255, 0, 255).astype(np.uint8)
            else:
                img_array = (SH2RGB(img_array) * 255).astype(np.uint8)

            # 将 NumPy 数组转换为 Pillow 图像
            img_array = Image.fromarray(img_array)
            img_path_group = f"{img_path}_{idx}.png"
            img_array.save(img_path_group)