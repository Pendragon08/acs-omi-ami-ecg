"""Author CNN (MyNet6View) from ACS-Baseline-main/models/model.py.

"Multi-View Multi-Scale Neural Network" for ECG classification.  Used in
Experiment 1 as the original-author OMI binary benchmark (num_classes=1).
Ported verbatim (including CoordAtt attention); wrapped to accept [B,1000,12].
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False)


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super().__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super().__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=16):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        mip = max(8, inp // reduction)
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = x.unsqueeze(2)
        identity = x
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = x.permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.act(self.bn1(self.conv1(y)))
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        out = identity * a_w * a_h
        return out.squeeze(2)


class Res2Block(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, kernel_size=5, stride=1, downsample=None,
                 groups=1, base_width=26, dilation=1, scale=4, first_block=True,
                 norm_layer=nn.BatchNorm1d, atten=True):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        width = int(planes * (base_width / 64.0)) * groups
        self.atten = atten
        self.conv1 = conv1x1(inplanes, width * scale)
        self.bn1 = norm_layer(width * scale)
        nb_branches = max(scale, 2) - 1
        if first_block:
            self.pool = nn.AvgPool1d(kernel_size=3, stride=stride, padding=1)
        self.convs = nn.ModuleList([
            nn.Conv1d(width, width, kernel_size=kernel_size, stride=stride,
                      padding=kernel_size // 2, groups=1, bias=False, dilation=1)
            for _ in range(nb_branches)
        ])
        self.bns = nn.ModuleList([norm_layer(width) for _ in range(nb_branches)])
        self.first_block = first_block
        self.scale = scale
        self.conv3 = conv1x1(width * scale, planes * self.expansion)
        self.relu = Mish()
        self.bn3 = norm_layer(planes * self.expansion)
        self.attention = CoordAtt(planes * self.expansion, planes * self.expansion) if atten else None
        self.shortcut = nn.Sequential()
        if stride != 1 or inplanes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv1d(inplanes, self.expansion * planes, kernel_size=1, stride=stride),
                nn.BatchNorm1d(self.expansion * planes),
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        xs = torch.chunk(out, self.scale, dim=1)
        y = 0
        for idx, conv in enumerate(self.convs):
            y = xs[idx] if self.first_block else y + xs[idx]
            y = conv(y)
            y = self.relu(self.bns[idx](y))
            out = torch.cat((out, y), 1) if idx > 0 else y
        if self.scale > 1:
            if self.first_block:
                out = torch.cat((out, self.pool(xs[len(self.convs)])), 1)
            else:
                out = torch.cat((out, xs[len(self.convs)]), 1)
        out = self.bn3(self.conv3(out))
        if self.attention is not None:
            out = self.attention(out)
        out += self.shortcut(x)
        return self.relu(out)


class MyNet(nn.Module):
    def __init__(self, num_classes=5, input_channels=12, single_view=False):
        super().__init__()
        self.single_view = single_view
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=25, padding=0, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = Mish()
        self.layer1 = Res2Block(inplanes=64, planes=128, kernel_size=15, stride=2, atten=True)
        self.layer2 = Res2Block(inplanes=128, planes=128, kernel_size=15, stride=2, atten=True)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        if not self.single_view:
            self.fc = nn.Linear(128, num_classes)

    def forward(self, x):  # x: [B, C, 1000]
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer2(self.layer1(out))
        out = self.avgpool(out).view(out.size(0), -1)
        if not self.single_view:
            out = self.fc(out)
        return out


class AdaptiveWeight(nn.Module):
    def __init__(self, planes=32):
        super().__init__()
        self.fc = nn.Linear(planes, 1)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        return self.sig(self.fc(x))


class MyNet6View(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.MyNet1 = MyNet(input_channels=1, single_view=True)
        self.MyNet2 = MyNet(input_channels=2, single_view=True)
        self.MyNet3 = MyNet(input_channels=2, single_view=True)
        self.MyNet4 = MyNet(input_channels=2, single_view=True)
        self.MyNet5 = MyNet(input_channels=2, single_view=True)
        self.MyNet6 = MyNet(input_channels=3, single_view=True)
        self.fuse_weights = nn.ModuleList([AdaptiveWeight(128) for _ in range(6)])
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):  # x: [B, 12, 1000] (standard lead order I,II,III,aVR,aVL,aVF,V1..V6)
        views = [
            self.MyNet1(x[:, 3, :].unsqueeze(1)),
            self.MyNet2(torch.cat((x[:, 0, :].unsqueeze(1), x[:, 4, :].unsqueeze(1)), dim=1)),
            self.MyNet3(x[:, 6:8, :]),
            self.MyNet4(x[:, 8:10, :]),
            self.MyNet5(x[:, 10:12, :]),
            self.MyNet6(torch.cat((x[:, 1:3, :], x[:, 5, :].unsqueeze(1)), dim=1)),
        ]
        out = sum(self.fuse_weights[i](v) * v for i, v in enumerate(views))
        return self.fc(out)


class MyNet6ViewModel(nn.Module):
    """Project-facing wrapper accepting [B, 1000, 12] (time-major)."""

    def __init__(self, num_classes=2):
        super().__init__()
        self.inner = MyNet6View(num_classes=num_classes)

    def forward(self, x, mask=None):
        return self.inner(x.transpose(1, 2))
