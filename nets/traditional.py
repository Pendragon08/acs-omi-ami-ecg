"""Traditional baselines (实施方案 section 16), ported from ACS-Baseline-main.

- LSTM (RNN)
- ResNet1D_Wang (CNN)
- InceptionTime (multi-scale CNN)
- ViT-1D (Transformer)

All bodies expect [B, 12, 1000] (channels-first) exactly as in the original
repo; the wrapper classes below accept the project's canonical [B, 1000, 12]
(time-major) input and transpose once.
"""
from functools import partial
from collections import OrderedDict

import torch
import torch.nn as nn


# --------------------------------------------------------------------------- #
# LSTM
# --------------------------------------------------------------------------- #
class RNN1d(nn.Sequential):
    def __init__(self, num_classes, input_channels=12, hidden_dim=256, num_layers=2,
                 bidirectional=False):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_channels, hidden_size=hidden_dim,
                            num_layers=num_layers, bidirectional=bidirectional)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(hidden_dim * (2 if bidirectional else 1), num_classes)

    def forward(self, x):  # x: [B, 12, 1000]
        x = x.transpose(1, 2).transpose(0, 1)   # [1000, B, 12]
        output = self.lstm(x)
        output = output[0].transpose(0, 1).transpose(1, 2)  # [B, H, 1000]
        output = self.avgpool(output).squeeze(-1)
        return self.fc(output)


class LSTM(nn.Module):
    def __init__(self, num_classes=2, **kwargs):
        super().__init__()
        self.inner = RNN1d(num_classes=num_classes, **kwargs)

    def forward(self, x):
        return self.inner(x.transpose(1, 2))


# --------------------------------------------------------------------------- #
# ResNet1D_Wang
# --------------------------------------------------------------------------- #
def _conv(in_planes, out_planes, stride=1, kernel_size=3):
    return nn.Conv1d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                     padding=(kernel_size - 1) // 2, bias=False)


class BasicBlock1d(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, kernel_size=[3, 3], downsample=None):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = [kernel_size, kernel_size // 2 + 1]
        self.conv1 = _conv(inplanes, planes, stride=stride, kernel_size=kernel_size[0])
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = _conv(planes, planes, kernel_size=kernel_size[1])
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        return self.relu(out)


class ResNet1d(nn.Sequential):
    def __init__(self, block, layers, kernel_size=5, num_classes=2, input_channels=12):
        super().__init__()
        self.inplanes = 64
        self.conv = nn.Conv1d(input_channels, 64, kernel_size=8, stride=2,
                              padding=7 // 2, bias=False)
        self.bn = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.block1 = self._make_layer(block, 64, layers[0], kernel_size=kernel_size)
        self.block2 = self._make_layer(block, 128, layers[1], stride=2, kernel_size=kernel_size)
        self.block3 = self._make_layer(block, 128, layers[1], stride=2, kernel_size=kernel_size)
        self.Avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1, kernel_size=3):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )
        layers = [block(self.inplanes, planes, stride, kernel_size, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):  # x: [B, 12, 1000]
        out = self.relu(self.bn(self.conv(x)))
        out = self.maxpool(out)
        out = self.block3(self.block2(self.block1(out)))
        out = self.Avgpool(out).view(out.size(0), -1)
        return self.fc(out)


class ResNet1D_Wang(nn.Module):
    def __init__(self, num_classes=2, **kwargs):
        super().__init__()
        self.inner = ResNet1d(BasicBlock1d, [1, 1, 1], num_classes=num_classes, **kwargs)

    def forward(self, x):
        return self.inner(x.transpose(1, 2))


# --------------------------------------------------------------------------- #
# InceptionTime
# --------------------------------------------------------------------------- #
class BaseBlock(nn.Module):
    def __init__(self, in_planes):
        super().__init__()
        self.bottleneck = nn.Conv1d(in_planes, 32, kernel_size=1, bias=False)
        self.conv4 = nn.Conv1d(32, 32, kernel_size=39, padding=19, bias=False)
        self.conv3 = nn.Conv1d(32, 32, kernel_size=19, padding=9, bias=False)
        self.conv2 = nn.Conv1d(32, 32, kernel_size=9, padding=4, bias=False)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        self.conv1 = nn.Conv1d(in_planes, 32, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm1d(128)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.bottleneck(x)
        o4 = self.conv4(out)
        o3 = self.conv3(out)
        o2 = self.conv2(out)
        o1 = self.conv1(self.maxpool(x))
        return self.relu(self.bn(torch.cat((o1, o2, o3, o4), dim=1)))


class InceptionTimeNet(nn.Module):
    def __init__(self, in_channel=12, num_classes=2):
        super().__init__()
        self.BaseBlock1 = BaseBlock(in_channel)
        self.BaseBlock2 = BaseBlock(128)
        self.BaseBlock3 = BaseBlock(128)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv1d(in_channel, 128, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(128)
        self.BaseBlock4 = BaseBlock(128)
        self.BaseBlock5 = BaseBlock(128)
        self.BaseBlock6 = BaseBlock(128)
        self.conv2 = nn.Conv1d(128, 128, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm1d(128)
        self.Avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):  # x: [B, 12, 1000]
        shortcut1 = self.bn1(self.conv1(x))
        o1 = self.BaseBlock3(self.BaseBlock2(self.BaseBlock1(x)))
        o1 = self.relu(o1 + shortcut1)
        shortcut2 = self.bn2(self.conv2(o1))
        o2 = self.BaseBlock6(self.BaseBlock5(self.BaseBlock4(o1)))
        o2 = self.relu(o2 + shortcut2)
        out = self.Avgpool(o2).view(o2.size(0), -1)
        return self.fc(out)


class InceptionTime(nn.Module):
    def __init__(self, num_classes=2, **kwargs):
        super().__init__()
        self.inner = InceptionTimeNet(in_channel=12, num_classes=num_classes, **kwargs)

    def forward(self, x):
        return self.inner(x.transpose(1, 2))


# --------------------------------------------------------------------------- #
# ViT-1D
# --------------------------------------------------------------------------- #
def _drop_path(x, drop_prob=0.0, training=False):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return _drop_path(x, self.drop_prob, self.training)


class PatchEmbed(nn.Module):
    def __init__(self, img_size=1000, patch_size=50, in_c=12, embed_dim=600):
        super().__init__()
        self.grid_size = (1, img_size // patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=(1, patch_size),
                              stride=(1, patch_size))
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return self.norm(x)


class Attention(nn.Module):
    def __init__(self, dim, num_heads=12, qkv_bias=True, qk_scale=None,
                 attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.proj(x))


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return self.drop(self.fc2(self.act(self.fc1(x))))


class ViTBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=True, qk_scale=None,
                 drop=0.0, attn_drop=0.0, drop_path=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads, qkv_bias, qk_scale, attn_drop, drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class VisionTransformer(nn.Module):
    def __init__(self, img_size=1000, patch_size=50, in_c=12, num_classes=2,
                 embed_dim=600, depth=2, num_heads=12, mlp_ratio=4.0, qkv_bias=True,
                 representation_size=512, drop=0.0, attn_drop=0.0, drop_path=0.0):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_c, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop)
        dpr = [x.item() for x in torch.linspace(0, drop_path, depth)]
        self.blocks = nn.Sequential(*[
            ViTBlock(embed_dim, num_heads, mlp_ratio, qkv_bias, None, drop, attn_drop, dpr[i])
            for i in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.pre_logits = nn.Sequential(OrderedDict([
            ("fc", nn.Linear(embed_dim, representation_size)),
            ("act", nn.Tanh()),
        ])) if representation_size else nn.Identity()
        self.head = nn.Linear(representation_size if representation_size else embed_dim, num_classes)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.01)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):  # x: [B, 12, 1000]
        x = self.patch_embed(x.unsqueeze(2))
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        x = self.blocks(x)
        x = self.norm(x)[:, 0]
        x = self.pre_logits(x)
        return self.head(x)


class ViT(nn.Module):
    def __init__(self, num_classes=2, **kwargs):
        super().__init__()
        self.inner = VisionTransformer(num_classes=num_classes, **kwargs)

    def forward(self, x):
        return self.inner(x.transpose(1, 2))
