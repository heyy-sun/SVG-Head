import os

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

def positional_encoding(tensor, num_encoding_functions=6, include_input=True, log_sampling=True):   # 为输入的 tensor 添加位置编码
    r"""Apply positional encoding to the input.
    Args:
        tensor (torch.Tensor): Input tensor to be positionally encoded. B x C x ...
        encoding_size (optional, int): Number of encoding functions used to compute
            a positional encoding (default: 6).
        include_input (optional, bool): Whether or not to include the input in the
            positional encoding (default: True).
    Returns:
    (torch.Tensor): Positional encoding of the input tensor.
    """
    if num_encoding_functions == 0:
        return tensor
    # TESTED
    # Trivially, the input tensor is added to the positional encoding.
    encoding = [tensor] if include_input else []
    frequency_bands = None
    if log_sampling:
        frequency_bands = 2.0 ** torch.linspace(
            0.0,
            num_encoding_functions - 1,
            num_encoding_functions,
            dtype=tensor.dtype,
            device=tensor.device,
        )
    else:
        frequency_bands = torch.linspace(
            2.0**0.0,
            2.0 ** (num_encoding_functions - 1),
            num_encoding_functions,
            dtype=tensor.dtype,
            device=tensor.device,
        )

    for freq in frequency_bands:
        for func in [torch.sin, torch.cos]:
            encoding.append(func(tensor * freq))

    # Special case, for no positional encoding
    if len(encoding) == 1:
        return encoding[0]
    else:
        return torch.cat(encoding, dim=1)

class NeuralTexture(nn.Module):
    def __init__(self, tex_size, tex_ch, mode, neural=True, init_value=None):
        super(NeuralTexture, self).__init__()

        self.neural = neural                            # 判断是否使用 neural texture
        self.tex_size = tex_size                        # 纹理尺寸
        self.tex_ch = tex_ch if self.neural else 3      # 纹理通道数
        self.mode = mode                                # 纹理编码模式（deep_prior or texture）

        self.get_texture_init = getattr(self, "{}_init".format(self.mode))  # 初始化方法
        self.get_texture = getattr(self, "{}_forward".format(self.mode))    # forward 方法
        assert not (
            self.get_texture_init is None or self.get_texture is None
        ), "No function match for {}_init or {}".format(self.mode)

        self.get_texture_init(init_value)

    def deep_prior_init(self, init_value = None):
        mgrid = torch.stack(torch.meshgrid([torch.linspace(-1, 1, self.tex_size)] * 2))[None].cuda()    # 创建一个二维网格 mgrid，范围是[-1, 1]
        self.mgrid = positional_encoding(mgrid, 6)          # 对 mgrid 进行位置编码  
        self.texture_decoder = UNet(self.mgrid.shape[1], self.tex_ch, bilinear=True)        # 创建一个 UNet 模型，输入通道数为 mgrid 的通道数，输出通道数为纹理通道数

    def deep_prior_forward(self):
        return self.texture_decoder(self.mgrid)             # 返回 texture_decoder 的输出

    def texture_init(self, init_value = None):
        if init_value == None:
            self.texture = nn.Parameter(
                torch.FloatTensor(1, self.tex_ch, self.tex_size, self.tex_size).uniform_(-1, 1), requires_grad=True   # 创建一个可训练的 texture
            )
        elif not torch.is_tensor(init_value):
            raise TypeError("init_value is not tensor")
        elif not init_value.shape == torch.Size([1, self.tex_ch, self.tex_size, self.tex_size]):
            raise ValueError("init_value.shape is wrong")
        else:
            self.texture = nn.Parameter(
                init_value, requires_grad=True   # 创建一个可训练的 texture
            )

    def texture_forward(self):
        return self.texture if self.neural else F.sigmoid(self.texture)   # 返回 texture（neural） 或者 texture 的 sigmoid

    def forward(self):
        return self.get_texture()


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_channels, out_channels))

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class DynamicDecoder(nn.Module):
    def __init__(self):
        super(DynamicDecoder, self).__init__()

        # self.neural = cfg.get("training.neural_texture", True)
        z_dim = 100
        outdim = 3
        self.z_fc = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.LeakyReLU(0.2, inplace=True),
        )

        base = 8
        self.upsample = nn.Sequential(
            ConvUpBlock(4, 64, base),
            ConvUpBlock(64, 32, base * 2),
            ConvUpBlock(32, 32, base * 4),
            ConvUpBlock(32, 16, base * 8),
            ConvUpBlock(16, 16, base * 16),
            ConvUpBlock(16, 8, base * 32),
            ConvUpBlock(8, outdim, base * 64),
        )

    def forward(self, z):
        B, _ = z.shape

        z_code = self.z_fc(z)
        z_code = z_code.view(B, 4, 8, 8)  # [B, 256] --> [B, 4, 8, 8]
        return self.upsample(z_code)

class Conv2dBNUB(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        feature_size,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=False,
    ):
        super(Conv2dBNUB, self).__init__()
        if isinstance(feature_size, int):
            feature_size = (feature_size, feature_size)
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=False,
        )
        self.bias = nn.Parameter(torch.zeros(out_channels, feature_size[0], feature_size[1]))

        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.bn(self.conv(x) + self.bias[None, ...])
        return out

# A Conv Upsample Block
class ConvUpBlock(nn.Module):
    def __init__(self, cin, cout, feature_size):
        super(ConvUpBlock, self).__init__()

        # self.conv1 = Conv2dWNUB(cin, cout, feature_size, 3, padding=1)
        self.conv1 = Conv2dBNUB(cin, cout, feature_size, 3, padding=1)
        self.conv2 = nn.Conv2d(cout, 4 * cout, 1)

        self.relu = nn.LeakyReLU(0.2)

        self.ps = nn.PixelShuffle(2)

    def forward(self, x):
        return self.ps(self.relu(self.conv2(self.conv1(x))))