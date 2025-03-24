import torch
import torch.nn as nn
from models.backbone_8stride import resnet18
import torchvision.models as models
import torch.nn.functional as F


class CrissCrossAttention(nn.Module):
    def __init__(self, in_channels, num_heads=2):
        super().__init__()
        self.num_heads = num_heads
        self.qkv_conv = nn.Conv2d(in_channels, in_channels * 3, 1, bias=False)
        self.out_conv = nn.Conv2d(in_channels, in_channels, 1, bias=False)
        self.softmax = nn.Softmax(dim=3)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.shape
        qkv = self.qkv_conv(x).chunk(3, dim=1)
        q, k, v = qkv  # Each [B, C//num_heads, H, W]

        # Reshape for multi-head attention
        q = q.view(B, self.num_heads, C // self.num_heads, H * W)
        k = k.view(B, self.num_heads, C // self.num_heads, H * W)
        v = v.view(B, self.num_heads, C // self.num_heads, H * W)

        # Horizontal attention
        energy_h = torch.matmul(q.permute(0, 1, 3, 2), k)  # [B, heads, H*W, H*W]
        attention_h = self.softmax(energy_h)
        out_h = torch.matmul(v, attention_h.permute(0, 1, 3, 2))  # [B, heads, C//h, H*W]

        # Vertical attention
        energy_v = torch.matmul(q, k.permute(0, 1, 3, 2))  # [B, heads, H*W, H*W]
        attention_v = self.softmax(energy_v)
        out_v = torch.matmul(attention_v, v.permute(0, 1, 3, 2))  # [B, heads, H*W, C//h]

        # Merge heads and reshape
        out = (out_h + out_v).view(B, C, H, W)
        out = self.gamma * self.out_conv(out) + x
        return out


class CCNet(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super().__init__()
        resnet = resnet18(pretrained=pretrained)
        self.backbone = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4
        )  # [B, 512, 16, 16]

        self.cca1 = CrissCrossAttention(512, num_heads=2)
        self.cca2 = CrissCrossAttention(512, num_heads=2)
        self.cls_head = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1),
            nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        )

    def forward(self, x):
        x = self.backbone(x)  # [B, 512, 16, 16]
        x = self.cca1(x)
        x = self.cca2(x)  # 两次交叉注意力迭代
        x = self.cls_head(x)  # [B, num_classes, 128, 128]
        return x