import torch
from torch import nn
import pandas as pd
import torch.nn.functional as F
from models.res_block import Res_Block, resblock
from models.xlstm_new import xLSTM
from models.multi_self_attention import MultiHeadAttention


class Mish(nn.Module):  # 定义一个继承自 nn.Module 的自定义激活函数类 Mish
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


class AdaptiveWeight(nn.Module):  # 定义一个继承自 nn.Module 的自适应权重类
    def __init__(self, plances=32):  # 初始化函数，定义网络结构，默认输入特征维度为 32
        super(AdaptiveWeight, self).__init__()

        self.fc1 = nn.Linear(plances, 1)  # 定义一个全连接层，将输入特征数 plances 映射到 1
        # self.bn = nn.BatchNorm1d(1)
        self.sig = nn.Sigmoid()  # 定义 Sigmoid 激活函数

    def forward(self, x):  # 定义前向传播函数
        out = self.fc1(x)
        # out = self.bn(out)
        out = self.sig(out)  # 经过 Sigmoid 激活函数，将输出缩放到 (0, 1) 范围内
        # out = out.view(x.size(0), 1, -1)

        return out


class Net_11(nn.Module):
    def __init__(self, input_size, num_classes=5, batch_first=True, num_lead=12, single_view=False):
        super(Net_11, self).__init__()

        self.single_view = single_view
        self.num_classes = num_classes
        self.num_lead = num_lead
        self.avg = nn.AdaptiveAvgPool1d(12)

        self.sep_conv12 = nn.Sequential(
            nn.Conv1d(in_channels=12, out_channels=12, kernel_size=25, stride=1, padding=0, bias=False),
            nn.BatchNorm1d(12),
            Mish(),
            resblock(input_size=input_size, inplanes=12, planes=32, kernel_size=5, stride=2, atten=False),
            nn.BatchNorm1d(32),
            Mish(),
            nn.Dropout(0.2),
            resblock(input_size=input_size, inplanes=32, planes=64, kernel_size=5, stride=2, atten=False),
            nn.BatchNorm1d(64),
            Mish(),
            nn.Dropout(0.2),
            resblock(input_size=input_size, inplanes=64, planes=64, kernel_size=5, stride=2, atten=False),
            nn.BatchNorm1d(64),
            Mish(),
            nn.Dropout(0.2),
            resblock(input_size=input_size, inplanes=64, planes=64, kernel_size=5, stride=2, atten=False),
            nn.AdaptiveAvgPool1d(12),
            # WTConv_resblock(input_size=320, inplanes=32, outplanes=64, kernel_size=15, stride=2, atten=True),
        )

        self.xlstm = xLSTM(input_size=488, head_size=5, num_heads=2, layers=['m', 's', 'm'],
                           batch_first=batch_first, proj_factor_slstm=4 / 3, proj_factor_mlstm=2)

        self.att = MultiHeadAttention(n_head=4,  # 多头数量
                                      d_model=64,  # 输入维度（特征维度）
                                      d_k=16,  # key 的维度
                                      d_v=16,  # value 的维度
                                      dropout=0.2)

        self.shortcut = nn.Sequential(
            nn.Conv1d(in_channels=12, out_channels=64, kernel_size=25, stride=2, padding=0, bias=False),
            nn.BatchNorm1d(64),
            Mish(),
        )

        if not self.single_view:
            self.fc = nn.Linear(12 * 64, num_classes)

    def forward(self, x):
        # print(f"x shape before sep_conv12: {x.shape}")
        # lead2 = x[:, self.select_lead - 1, :].view(x.size(0), 1, -1)
        lead12 = self.sep_conv12(x)

        shortcut_feature = self.shortcut(x)
        xlstm_feature, _ = self.xlstm(shortcut_feature)

        shortcut_feature1 = self.avg(shortcut_feature)

        xlstm_feature += shortcut_feature
        xlstm_feature = self.avg(xlstm_feature)

        lead12 = lead12.permute(0, 2, 1)  # 调整为 (batch_size, seq_len, channels)
        xlstm_feature = xlstm_feature.permute(0, 2, 1)  # 调整为 (batch_size, seq_len, channels)
        shortcut_feature1 = shortcut_feature1.permute(0, 2, 1)  # 调整为 (batch_size, seq_len, channels)

        attention_output, _ = self.att(lead12, xlstm_feature,
                                       shortcut_feature1)  # 输出形状: (batch_size, seq_len, channels)
        attention_output, _ = self.att(attention_output, attention_output, attention_output)

        fusion = attention_output.reshape(attention_output.size(0), -1)  # 展平为 (batch_size, channels * seq_len)
        if not self.single_view:  # 如果 single_view 为 False，进行分类
            fusion = self.fc(fusion)  # 经过全连接层输出类别

        return fusion

def net_11(num_classes=5, input_size=1000, num_lead=12, single_view=False):
    return Net_11(num_classes=num_classes, input_size=input_size, num_lead=num_lead, single_view=single_view)
