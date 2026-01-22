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
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def truncated_l1_loss(pred, gt, reduction="mean", beta=1.0, rate=1.0):
    delta_map = torch.abs(pred - gt)
    mask = delta_map > beta
    if mask.sum() == 0:
        return torch.tensor(0.0).cuda()

    loss = rate * (delta_map[mask] - beta)
    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()
    else:
        raise NotImplementedError

    return loss

def gradient_smooth(posMap, method="smooth_l1", beta=8):
    """Compute the gradient smooth item

    Args:
        posMap (float): [B, 3, 256, 256]

    Returns:
        _type_: _description_
    """
    if method == "l1":
        diff_x = torch.abs(posMap[:, :, 1:, :] - posMap[:, :, :-1, :]).mean()
        diff_y = torch.abs(posMap[..., 1:] - posMap[..., :-1]).mean()
    elif method == "l2":
        diff_x = F.mse_loss(posMap[:, :, 1:, :], posMap[:, :, :-1, :], reduction="mean")
        diff_y = F.mse_loss(posMap[..., 1:], posMap[..., :-1], reduction="mean")
    elif method == "smooth_l1":
        diff_x = F.smooth_l1_loss(posMap[:, :, 1:, :], posMap[:, :, :-1, :], reduction="mean", beta=beta)
        diff_y = F.smooth_l1_loss(posMap[..., 1:], posMap[..., :-1], reduction="mean", beta=beta)
    elif method == "truncated_l1":
        diff_x = truncated_l1_loss(posMap[:, :, 1:, :], posMap[:, :, :-1, :], reduction="mean", beta=beta)
        diff_y = truncated_l1_loss(posMap[..., 1:], posMap[..., :-1], reduction="mean", beta=beta)
    else:
        raise NotImplementedError

    return diff_x + diff_y

def alpha_loss(alpha_map):
    non_zero_mask = (alpha_map > 0)
    diff = alpha_map.sub(1)
    diff_squared = diff * diff
    loss = torch.sum(non_zero_mask * diff_squared)

    return loss