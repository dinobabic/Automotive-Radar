import torch
import torch.nn as nn

import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))
    
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = ConvBlock(in_channels, out_channels, kernel_size=3, stride=2, padding=1)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2),
            nn.BatchNorm2d(out_channels)
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = self.downsample(x)
        out = self.conv1(x)
        out = self.bn2(self.conv2(out))
        out += identity
        return self.relu(out)

class Encoder(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.layer1 = ConvBlock(in_channels, 64, kernel_size=7, stride=2, padding=3)
        self.layer2 = ResidualBlock(64, 128)
        self.layer3 = ResidualBlock(128, 256)
        self.layer4 = ResidualBlock(256, 512)
        self.layer5 = ResidualBlock(512, 1024)

    def forward(self, x):
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x5 = self.layer5(x4)
        return x5, [x1, x2, x3, x4]

"""
    NVRadarNet like architecture for radar-based object detection.
"""
class NVRadarNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()

        self.encoder = Encoder(in_channels)

        # class segmentation head - for each pixel predicts class probabilites
        self.class_seg_upsample_1 = nn.ConvTranspose2d(1024, 256, kernel_size=4, stride=4)
        self.class_seg_upsample_2 = nn.ConvTranspose2d(256, 64, kernel_size=4, stride=4)
        self.class_seg = nn.Conv2d(64, num_classes, kernel_size=1, stride=1)

        # regression head
        self.reg_upsample_1 = nn.ConvTranspose2d(1024, 256, kernel_size=4, stride=4)
        self.reg_upsample_2 = nn.ConvTranspose2d(256, 64, kernel_size=4, stride=4)
        self.reg_out = nn.Conv2d(64, 6, kernel_size=1)

        # bayesian learned weights for loss balancing
        self.log_var_seg = nn.Parameter(torch.zeros(1))
        self.log_var_reg = nn.Parameter(torch.tensor([2.5]))

    def forward(self, x):
        x5, skips = self.encoder(x)
        
        seg = self.class_seg_upsample_1(x5)
        skip1 = F.interpolate(skips[2], size=seg.shape[-2:], mode='bilinear', align_corners=False)
        seg = seg + skip1

        seg = self.class_seg_upsample_2(seg)
        skip2 = F.interpolate(skips[0], size=seg.shape[-2:], mode='bilinear', align_corners=False)
        seg = seg + skip2
        seg = self.class_seg(seg)


        reg = self.reg_upsample_1(x5)
        reg = reg + skip1

        reg = self.reg_upsample_2(reg)
        reg = reg + skip2
        reg = self.reg_out(reg)

        return seg, reg
