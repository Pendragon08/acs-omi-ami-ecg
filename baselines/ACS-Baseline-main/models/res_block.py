import torch
from torch import nn
from models.attention import CoordAtt
from models.xlstm_new import xLSTM
import torch.nn.functional as F

class Mish(nn.Module):   # 定义一个继承自 nn.Module 的自定义激活函数类 Mish
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * (torch.tanh(F.softplus(x)))  # 返回 Mish 激活函数的计算结果：x * tanh(softplus(x))


def conv1x1(in_planes, out_planes, stride=1):  # 定义一个函数 conv1x1，输入通道数、输出通道数和步幅
    return nn.Conv1d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     padding=0,
                     bias=False
                     )

class Res_Block(nn.Module):  # 定义 Res2Block 类，继承自 nn.Module
    expansion = 1   # 定义膨胀倍数，默认为 1

    # 初始化函数，传入参数包括输入输出通道数、卷积核大小、步幅、分组数、基础宽度、扩展比例、是否为第一个block、归一化层类型、是否使用注意力机制
    def __init__(self, input_size, inplanes, planes, kernel_size=5, stride=1, downsample=None, batch_first=True,
                 groups=1, base_width=26, dilation=1, scale=4, first_block=True, norm_layer=nn.BatchNorm1d,
                 atten=True):

        super(Res_Block, self).__init__()
        if norm_layer is None:   # 如果没有指定归一化层，默认为 nn.BatchNorm1d
            norm_layer = nn.BatchNorm1d

        width = int(planes * (base_width / 64.)) * groups     # 根据 base_width 和 groups 计算卷积的宽度
        # print(width)

        self.atten = atten

        self.conv1 = conv1x1(inplanes, width * scale)   # 定义第一个 1x1 卷积，改变通道数
        self.bn1 = norm_layer(width * scale)    # 定义第一个批归一化层

        # If scale == 1, single conv else identity & (scale - 1) convs
        nb_branches = max(scale, 2) - 1    # 计算分支数量（最少为1个分支）
        if first_block:    # 如果是第一个block，使用平均池化来下采样
            self.pool = nn.AvgPool1d(kernel_size=3, stride=stride, padding=1)
        # 定义多个卷积层，用于多分支计算
        self.convs = nn.ModuleList([nn.Conv1d(width, width, kernel_size=kernel_size, stride=stride,
                                              padding=kernel_size // 2, groups=1, bias=False, dilation=1)
                                    for _ in range(nb_branches)])
        self.bns = nn.ModuleList([norm_layer(width) for _ in range(nb_branches)])   # 定义对应的批归一化层
        self.first_block = first_block   # 标记是否为第一个block
        self.scale = scale   # scale 的值决定有多少分支

        self.conv3 = conv1x1(width * scale, planes * self.expansion)   # 定义第二个 1x1 卷积层

        self.relu = Mish()   # 定义 Mish 激活函数
        self.bn3 = norm_layer(planes * self.expansion)  # bn reverse   # 定义最后的批归一化层

        # self.dropout = nn.Dropout(.1)

        if self.atten is True:    # 如果使用注意力机制
            # self.attention = SELayer(planes * self.expansion)
            # self.attention = CBAM(planes * self.expansion)
            self.attention = CoordAtt(planes * self.expansion, planes * self.expansion)   # 使用CoordAtt注意力机制
        else:  # 不使用注意力机制
            self.attention = None

        self.shortcut = nn.Sequential()   # 定义捷径连接，用于跳跃连接
        if stride != 1 or inplanes != self.expansion * planes:    # 如果需要匹配输入输出的通道数或步幅
            self.shortcut = nn.Sequential(
                nn.Conv1d(inplanes, self.expansion * planes, kernel_size=1, stride=stride),
                nn.BatchNorm1d(self.expansion * planes),
                xLSTM(input_size=input_size,
                           head_size=16,
                           num_heads=4,
                           layers=['s', 'm'],
                           batch_first=batch_first,
                           proj_factor_slstm=4 / 3,
                           proj_factor_mlstm=2)
            )
            self.shortcut_add = nn.Sequential(
                nn.Conv1d(inplanes, self.expansion * planes, kernel_size=1, stride=stride),
                nn.BatchNorm1d(self.expansion * planes),
            )

    def forward(self, x):
        # print(f"x shape before conv1 in res_block:{x.shape}")

        out = self.conv1(x)


        out = self.relu(out)
        out = self.bn1(out)  # bn reverse

        # Chunk the feature map
        xs = torch.chunk(out, self.scale, dim=1)   # 将输入在通道维度上分块（chunk）
        # print(xs)

        # Initialize output as empty tensor for proper concatenation
        y = 0  # 初始化 y，用于拼接
        for idx, conv in enumerate(self.convs):   # 遍历每个分支的卷积层
            # Add previous y-value
            if self.first_block:  # 如果是第一个 block，直接使用分块结果
                y = xs[idx]
            else:   # 否则进行累加
                y += xs[idx]
            y = conv(y)   # 卷积运算
            y = self.relu(self.bns[idx](y))
            # Concatenate with previously computed values
            out = torch.cat((out, y), 1) if idx > 0 else y   # 将每次的结果拼接到一起
        # Use last chunk as x1
        if self.scale > 1:   # 如果 scale 大于1，拼接最后一块
            if self.first_block:
                out = torch.cat((out, self.pool(xs[len(self.convs)])), 1)    # 如果是第一个block，池化最后一个块
            else:
                out = torch.cat((out, xs[len(self.convs)]), 1)   # 否则直接拼接

        # out = self.dropout(out)

        out = self.conv3(out)   # 第二个 1x1 卷积
        out = self.bn3(out)   # 批归一化

        if self.atten:    # 如果有注意力机制
            out = self.attention(out)   # 施加注意力机制
        x_lstm, _ = self.shortcut(x)
        out += x_lstm   # 加上 shortcut 的输出，实现残差连接

        out = self.shortcut_add(x)
        out += out

        out = self.relu(out)

        return out

class resblock(nn.Module):  # 定义 Res2Block 类，继承自 nn.Module
    expansion = 1   # 定义膨胀倍数，默认为 1

    # 初始化函数，传入参数包括输入输出通道数、卷积核大小、步幅、分组数、基础宽度、扩展比例、是否为第一个block、归一化层类型、是否使用注意力机制
    def __init__(self, input_size, inplanes, planes, kernel_size=5, stride=1, downsample=None, batch_first=True,
                 groups=1, base_width=26, dilation=1, scale=4, first_block=True, norm_layer=nn.BatchNorm1d,
                 atten=True):

        super(resblock, self).__init__()
        if norm_layer is None:   # 如果没有指定归一化层，默认为 nn.BatchNorm1d
            norm_layer = nn.BatchNorm1d

        width = int(planes * (base_width / 64.)) * groups     # 根据 base_width 和 groups 计算卷积的宽度
        # print(width)

        self.atten = atten

        self.conv1 = conv1x1(inplanes, width * scale)   # 定义第一个 1x1 卷积，改变通道数
        self.bn1 = norm_layer(width * scale)    # 定义第一个批归一化层

        # If scale == 1, single conv else identity & (scale - 1) convs
        nb_branches = max(scale, 2) - 1    # 计算分支数量（最少为1个分支）
        if first_block:    # 如果是第一个block，使用平均池化来下采样
            self.pool = nn.AvgPool1d(kernel_size=3, stride=stride, padding=1)
        # 定义多个卷积层，用于多分支计算
        self.convs = nn.ModuleList([nn.Conv1d(width, width, kernel_size=kernel_size, stride=stride,
                                              padding=kernel_size // 2, groups=1, bias=False, dilation=1)
                                    for _ in range(nb_branches)])
        self.bns = nn.ModuleList([norm_layer(width) for _ in range(nb_branches)])   # 定义对应的批归一化层
        self.first_block = first_block   # 标记是否为第一个block
        self.scale = scale   # scale 的值决定有多少分支

        self.conv3 = conv1x1(width * scale, planes * self.expansion)   # 定义第二个 1x1 卷积层

        self.relu = Mish()   # 定义 Mish 激活函数
        self.bn3 = norm_layer(planes * self.expansion)  # bn reverse   # 定义最后的批归一化层

        # self.dropout = nn.Dropout(.1)

        if self.atten is True:    # 如果使用注意力机制
            # self.attention = SELayer(planes * self.expansion)
            # self.attention = CBAM(planes * self.expansion)
            self.attention = CoordAtt(planes * self.expansion, planes * self.expansion)   # 使用CoordAtt注意力机制
        else:  # 不使用注意力机制
            self.attention = None

        self.shortcut = nn.Sequential()   # 定义捷径连接，用于跳跃连接
        if stride != 1 or inplanes != self.expansion * planes:    # 如果需要匹配输入输出的通道数或步幅
            self.shortcut = nn.Sequential(
                nn.Conv1d(inplanes, self.expansion * planes, kernel_size=1, stride=stride),
                nn.BatchNorm1d(self.expansion * planes),
            )


    def forward(self, x):
        # print(f"x shape before conv1 in res_block:{x.shape}")

        out = self.conv1(x)


        out = self.relu(out)
        out = self.bn1(out)  # bn reverse

        # Chunk the feature map
        xs = torch.chunk(out, self.scale, dim=1)   # 将输入在通道维度上分块（chunk）

        # Initialize output as empty tensor for proper concatenation
        y = 0  # 初始化 y，用于拼接
        for idx, conv in enumerate(self.convs):   # 遍历每个分支的卷积层
            # Add previous y-value
            if self.first_block:  # 如果是第一个 block，直接使用分块结果
                y = xs[idx]
            else:   # 否则进行累加
                y += xs[idx]
            y = conv(y)   # 卷积运算
            y = self.relu(self.bns[idx](y))
            # Concatenate with previously computed values
            out = torch.cat((out, y), 1) if idx > 0 else y   # 将每次的结果拼接到一起
        # Use last chunk as x1
        if self.scale > 1:   # 如果 scale 大于1，拼接最后一块
            if self.first_block:
                out = torch.cat((out, self.pool(xs[len(self.convs)])), 1)    # 如果是第一个block，池化最后一个块
            else:
                out = torch.cat((out, xs[len(self.convs)]), 1)   # 否则直接拼接

        # out = self.dropout(out)

        out = self.conv3(out)   # 第二个 1x1 卷积
        out = self.bn3(out)   # 批归一化

        if self.atten:    # 如果有注意力机制
            out = self.attention(out)   # 施加注意力机制


        out += self.shortcut(x)

        out = self.relu(out)

        return out