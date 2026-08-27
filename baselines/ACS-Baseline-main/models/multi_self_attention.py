from torch import nn
import torch

class ScaledDotProductAttention(nn.Module):
    """ Scaled Dot-Product Attention """

    def __init__(self, scale_factor, dropout=0.0):
        super().__init__()
        self.scale_factor = scale_factor
        # dropout用于防止过拟合，在前向传播的过程中，让某个神经元的激活值以一定的概率停止工作
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        # batch_size: 批量大小
        # len_q,len_k,len_v: 序列长度 在这里他们都相等
        # n_head: 多头注意力，论文中默认为8
        # d_k,d_v: k v 的dim(维度) 默认都是64
        # 此时q的shape为(batch_size, n_head, len_q, d_k) (batch_size, 8, len_q, 64)
        # 此时k的shape为(batch_size, n_head, len_k, d_k) (batch_size, 8, len_k, 64)
        # 此时v的shape为(batch_size, n_head, len_k, d_v) (batch_size, 8, len_k, 64)
        # q先除以self.scale_factor，再乘以k的转置(交换最后两个维度(这样才可以进行矩阵相乘))。
        # attn的shape为(batch_size, n_head, len_q, len_k)
        attn = torch.matmul(q / self.scale_factor, k.transpose(2, 3))
        # if mask is not None:
        #     """
        #     用-1e9代替0 -1e9是一个很大的负数 经过softmax之后接近0
        #     # 其一：去除掉各种padding在训练过程中的影响
        #     # 其二，将输入进行遮盖，避免decoder看到后面要预测的东西。（只用在decoder中）
        #     """
        #     scores = scores.masked_fill(mask == 0, -1e9)
        # 先在attn的最后一个维度做softmax 再dropout 得到注意力分数
        attn = self.dropout(torch.softmax(attn, dim=-1))
        # 最后attn与v矩阵相乘
        # output的shape为(batch_size, 8, len_q, 64)
        output = torch.matmul(attn, v)
        # 返回 output和注意力分数
        return output, attn



class MultiHeadAttention(nn.Module):
    """ Multi-Head Attention module """
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        # 论文中这里的n_head, d_model, d_k, d_v分别默认为8, 512, 64, 64
        # q k v先经过不同的线性层，再用ScaledDotProductAttention，最后再经过一个线性层
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)
        self.attention = ScaledDotProductAttention(scale_factor=d_k ** 0.5)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)# 默认对最后一个维度初始化
    def forward(self, q, k, v, mask=None):
        # q, k, v初次输入为含位置信息的嵌入矩阵X，由于要堆叠N次，后面的输入则是上个多头的输出
        # q, k, v：batch_size * seq_num * d_model
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        # len_q, len_k, len_v 为输入的序列长度
        # batch_size为batch_size
        batch_size, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)
        # 用作残差连接
        residual = q
        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        # q k v 分别经过一个线性层再改变维度
        # 由(batch_size, len_q, n_head*d_k) => (batch_size, len_q, n_head, d_k)
        # (batch_size, len_q, 8*64) => (batch_size, len_q, 8, 64)
        q = self.layer_norm(q)
        k = self.layer_norm(k)
        v = self.layer_norm(v)
        # 与q,k,v相关矩阵相乘，得到相应的q,k,v向量，d_model=n_head * d_k
        q = self.w_qs(q).view(batch_size, len_q, n_head, d_k)
        k = self.w_ks(k).view(batch_size, len_k, n_head, d_k)
        v = self.w_vs(v).view(batch_size, len_v, n_head, d_v)
        # Transpose for attention dot product: b x n x lq x dv
        # 交换维度做attention
        # (batch_size, len_q, n_head, d_k) => (batch_size, n_head, len_q, d_k)
        # (batch_size, len_q, 8, 64) => (batch_size, 8, len_q, 64)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        if mask is not None:
            # 为head增加一个维度
            mask = mask.unsqueeze(1)   # For head axis broadcasting.
        # 输出的q为Softmax(QK/d + (1-S)σ)V, attn 为QK/D
        q, attn = self.attention(q, k, v, mask=mask)
        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        # (batch_size, 8, len_k, 64) => (batch_size, len_k, 8, 64) => (batch_size, len_k, 512)
        q = q.transpose(1, 2).contiguous().view(batch_size, len_q, -1)
        # 经过fc和dropout
        q = self.dropout(self.fc(q))
        # 残差连接 论文中的Add & Norm中的Add
        q += residual
        # 论文中的Add & Norm中的Norm
        q = self.layer_norm(q)
        # q的shape为(batch_size, len_q, 512)
        # attn的shape为(batch_size, n_head, len_q, len_k)
        return q, attn