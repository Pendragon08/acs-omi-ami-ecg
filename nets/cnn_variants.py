"""Extended CNN variants matching the *existing* baselines' scale exactly.

To make the family comparison a clean structural ablation, the four models below
reuse the channel / kernel scale of the existing baselines and only swap the
convolutional structure:

    ResNet1D-Wang  ->  SEResNet1D_Wang   (same 64/128/128 + k8/5, + SE block)
    ResNet1D-Wang  ->  Res2Net1D_Wang    (same scale, Res2Net multi-scale block)
    InceptionTime  ->  XceptionTime      (same 128ch/6 blocks/39-19-9, depthwise-separable)
    InceptionTime  ->  LITEMV            (same scale, depthwise-separable + dilated)

Bodies expect [B, 12, 1000] (channels-first); wrappers accept [B, 1000, 12].
"""
import torch
import torch.nn as nn


def _conv(in_planes, out_planes, stride=1, kernel_size=3):
    return nn.Conv1d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                     padding=(kernel_size - 1) // 2, bias=False)


# --------------------------------------------------------------------------- #
# SE block
# --------------------------------------------------------------------------- #
class SEBlock1d(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Conv1d(channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden, channels, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.fc(self.pool(x))


# --------------------------------------------------------------------------- #
# ResNet-family blocks (match ResNet1D_Wang: k=5, 64/128/128)
# --------------------------------------------------------------------------- #
class SEBasicBlock1d(nn.Module):
    """ResNet1D_Wang BasicBlock + SE (conv1 k5, conv2 k3)."""

    expansion = 1

    def __init__(self, inplanes, planes, stride=1, kernel_size=5, reduction=16):
        super().__init__()
        k2 = kernel_size // 2 + 1                     # 3
        self.conv1 = _conv(inplanes, planes, stride, kernel_size)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = _conv(planes, planes, 1, k2)
        self.bn2 = nn.BatchNorm1d(planes)
        self.se = SEBlock1d(planes, reduction)
        self.downsample = None
        if stride != 1 or inplanes != planes:
            self.downsample = nn.Sequential(
                nn.Conv1d(inplanes, planes, 1, stride=stride, bias=False),
                nn.BatchNorm1d(planes),
            )

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)


class Res2Block1d(nn.Module):
    """Res2Net multi-scale residual block (hierarchical group-wise conv, k=5)."""

    def __init__(self, inplanes, planes, stride=1, kernel_size=5, scale=4):
        super().__init__()
        self.scale = scale
        self.width = planes // scale
        self.conv1 = nn.Conv1d(inplanes, planes, 1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.convs = nn.ModuleList([
            _conv(self.width, self.width, 1, kernel_size)
            for _ in range(scale - 1)
        ])
        self.bns = nn.ModuleList(
            [nn.BatchNorm1d(self.width) for _ in range(scale - 1)])
        self.conv3 = nn.Conv1d(planes, planes, 1, bias=False)
        self.bn3 = nn.BatchNorm1d(planes)
        self.downsample = None
        if stride != 1 or inplanes != planes:
            self.downsample = nn.Sequential(
                nn.Conv1d(inplanes, planes, 1, stride=stride, bias=False),
                nn.BatchNorm1d(planes),
            )

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        xs = torch.split(out, self.width, dim=1)
        y = xs[0]
        outs = [y]
        for i in range(1, self.scale):
            y = self.relu(self.bns[i - 1](self.convs[i - 1](xs[i] + y)))
            outs.append(y)
        out = torch.cat(outs, dim=1)
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)


class _ResNetBackbone(nn.Module):
    """Mirror of ResNet1d in traditional.py (k8 stem, 64/128/128, k5 blocks)."""

    def __init__(self, block, num_classes):
        super().__init__()
        self.conv = nn.Conv1d(12, 64, kernel_size=8, stride=2, padding=3, bias=False)
        self.bn = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.block1 = block(64, 64, stride=1, kernel_size=5)
        self.block2 = block(64, 128, stride=2, kernel_size=5)
        self.block3 = block(128, 128, stride=2, kernel_size=5)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):                           # x: [B, 12, 1000]
        out = self.relu(self.bn(self.conv(x)))
        out = self.maxpool(out)
        out = self.block3(self.block2(self.block1(out)))
        out = self.avgpool(out).view(out.size(0), -1)
        return self.fc(out)


# --------------------------------------------------------------------------- #
# Inception-family blocks (match InceptionTime: 32 bottleneck, 128 out, 6 blocks)
# --------------------------------------------------------------------------- #
class _DWSepConv1d(nn.Module):
    """Depthwise-separable conv: depthwise(k, groups=C) -> pointwise(1x1)."""

    def __init__(self, in_c, out_c, kernel_size, dilation=1):
        super().__init__()
        self.depthwise = nn.Conv1d(
            in_c, in_c, kernel_size, stride=1,
            padding=(kernel_size - 1) // 2 * dilation, dilation=dilation,
            groups=in_c, bias=False)
        self.pointwise = nn.Conv1d(in_c, out_c, 1, bias=False)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class XceptionBlock(nn.Module):
    """InceptionTime BaseBlock with depthwise-separable convs (k = 39/19/9)."""

    def __init__(self, in_c, kernels=(39, 19, 9)):
        super().__init__()
        self.bottleneck = nn.Conv1d(in_c, 32, 1, bias=False)
        self.sep = nn.ModuleList([_DWSepConv1d(32, 32, k) for k in kernels])
        self.maxpool = nn.MaxPool1d(3, stride=1, padding=1)
        self.pool_conv = nn.Conv1d(in_c, 32, 1, bias=False)
        self.bn = nn.BatchNorm1d(128)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        b = self.bottleneck(x)
        branches = [sep(b) for sep in self.sep]
        branches.append(self.pool_conv(self.maxpool(x)))
        return self.relu(self.bn(torch.cat(branches, dim=1)))


class LITEMVBlock(nn.Module):
    """Lightweight InceptionTime block: depthwise-separable + dilated multiscale.

    k = 9 with dilation 1/2/4 -> receptive fields 9/17/33 (~= InceptionTime 9/19/39).
    """

    def __init__(self, in_c, kernel=9, dilations=(1, 2, 4)):
        super().__init__()
        self.bottleneck = nn.Conv1d(in_c, 32, 1, bias=False)
        self.sep = nn.ModuleList(
            [_DWSepConv1d(32, 32, kernel, dilation=d) for d in dilations])
        self.maxpool = nn.MaxPool1d(3, stride=1, padding=1)
        self.pool_conv = nn.Conv1d(in_c, 32, 1, bias=False)
        self.bn = nn.BatchNorm1d(128)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        b = self.bottleneck(x)
        branches = [sep(b) for sep in self.sep]
        branches.append(self.pool_conv(self.maxpool(x)))
        return self.relu(self.bn(torch.cat(branches, dim=1)))


class _InceptionBackbone(nn.Module):
    """Mirror of InceptionTimeNet: 6 blocks @128ch, 2 residual shortcuts."""

    def __init__(self, block, num_classes):
        super().__init__()
        self.block1 = block(12)
        self.block2 = block(128)
        self.block3 = block(128)
        self.conv1 = nn.Conv1d(12, 128, 1, bias=False)
        self.bn1 = nn.BatchNorm1d(128)
        self.block4 = block(128)
        self.block5 = block(128)
        self.block6 = block(128)
        self.conv2 = nn.Conv1d(128, 128, 1, bias=False)
        self.bn2 = nn.BatchNorm1d(128)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, num_classes)

    def forward_features(self, x):                  # x: [B, 12, T] -> [B, 128]
        shortcut1 = self.bn1(self.conv1(x))
        o1 = self.block3(self.block2(self.block1(x)))
        o1 = self.relu(o1 + shortcut1)
        shortcut2 = self.bn2(self.conv2(o1))
        o2 = self.block6(self.block5(self.block4(o1)))
        o2 = self.relu(o2 + shortcut2)
        return self.avgpool(o2).view(o2.size(0), -1)

    def forward(self, x):                           # x: [B, 12, 1000]
        return self.fc(self.forward_features(x))


# --------------------------------------------------------------------------- #
# Public wrappers ([B, 1000, 12] -> channels-first)
# --------------------------------------------------------------------------- #
class SEResNet1D_Wang(nn.Module):
    def __init__(self, num_classes=2, **kwargs):
        super().__init__()
        self.inner = _ResNetBackbone(SEBasicBlock1d, num_classes=num_classes)

    def forward(self, x, mask=None):
        return self.inner(x.transpose(1, 2))


class Res2Net1D_Wang(nn.Module):
    def __init__(self, num_classes=2, **kwargs):
        super().__init__()
        self.inner = _ResNetBackbone(Res2Block1d, num_classes=num_classes)

    def forward(self, x, mask=None):
        return self.inner(x.transpose(1, 2))


class XceptionTime(nn.Module):
    def __init__(self, num_classes=2, **kwargs):
        super().__init__()
        self.inner = _InceptionBackbone(XceptionBlock, num_classes=num_classes)

    def forward(self, x, mask=None):
        return self.inner(x.transpose(1, 2))


class LITEMV(nn.Module):
    def __init__(self, num_classes=2, **kwargs):
        super().__init__()
        self.inner = _InceptionBackbone(LITEMVBlock, num_classes=num_classes)

    def forward(self, x, mask=None):
        return self.inner(x.transpose(1, 2))


class XceptionTimeFusion(nn.Module):
    """Multimodal XceptionTime: raw 10s waveform + median beat + clinical features.

    - raw  branch: _InceptionBackbone(XceptionBlock) -> [B,128] from [B,1000,12]
    - med  branch: _InceptionBackbone(XceptionBlock) -> [B,128] from [B,100,12]
      (1s median-averaged beat resampled to 100 Hz; same Xception encoder applies
      because 1D convs + global avg-pool are length-agnostic)
    - clin branch: 2-layer MLP -> [B,clin_hidden] from [B,clin_dim] (gender, age)

    Fused by concatenation + a linear head.  ``med``/``clin`` may be None (then
    the corresponding branch is zeroed) so the same class also serves as an
    ablation of a single branch.
    """

    def __init__(self, num_classes=2, clin_dim=2, clin_hidden=32, **kwargs):
        super().__init__()
        self.clin_dim = clin_dim
        self.raw_enc = _InceptionBackbone(XceptionBlock, num_classes=num_classes)
        self.raw_enc.fc = nn.Identity()             # feature extractor only
        self.med_enc = _InceptionBackbone(XceptionBlock, num_classes=num_classes)
        self.med_enc.fc = nn.Identity()
        self.clin_mlp = nn.Sequential(
            nn.Linear(clin_dim, clin_hidden), nn.ReLU(inplace=True),
            nn.Linear(clin_hidden, clin_hidden), nn.ReLU(inplace=True),
        )
        self.head = nn.Linear(128 + 128 + clin_hidden, num_classes)

    def forward(self, x, mask=None, med=None, clin=None):
        # x: [B,1000,12] (time-major); med: [B,100,12]; clin: [B,clin_dim]
        feat_raw = self.raw_enc.forward_features(x.transpose(1, 2))
        if med is None:
            feat_med = feat_raw.new_zeros(feat_raw.size(0), 128)
        else:
            feat_med = self.med_enc.forward_features(med.transpose(1, 2))
        if clin is None:
            clin = feat_raw.new_zeros(feat_raw.size(0), self.clin_dim)
        cf = self.clin_mlp(clin)
        return self.head(torch.cat([feat_raw, feat_med, cf], dim=1))
