# 这一部分实际上包括了两个概念，无mask多头注意力机制和有mask的多头注意力机制
'''
注意：d_model实际上是句子中的单词编码后的维度
'''
import math

import torch
from torch import nn
from torch.nn import functional as F



class MultiHeadAttention(nn.Module):
    def __init__(self,n_heads, d_model,dropout=0.1):
        super(MultiHeadAttention,self).__init__()
        self.n_heads=n_heads
        self.d_model=d_model

        self.q_l=nn.Linear(d_model,d_model)
        self.k_l=nn.Linear(d_model,d_model)
        self.v_l=nn.Linear(d_model,d_model)
        self.dropout=nn.Dropout(dropout)
        self.out =nn.Linear(d_model,d_model)

    def attention(self,q,k,v,d_k,mask=None,dropout=None):
        # torch.matmul在计算什么
        # 对于一维向量来说实际上就是一个点乘的情况
        # 对于二维矩阵实际上是矩阵相乘
        # 对于更高维的张量实际上只注重最后两个位置的值（实际上是个按位置的矩阵相乘）
        scores=torch.matmul(q,k.transpose(-2,-1))/math.sqrt(d_k)

        ## 掩码部分
        if mask is not None:
            mask=mask.unsqueeze(1)
            scores=scores.masked_fill(mask==0,-1e9)

        scores=F.softmax(scores,dim=-1)

        if dropout is not None:
            scores=dropout(scores)

        output=torch.matmul(scores,v)
        return output

    def forward(self,x_q,x_k,x_v,mask=None):
        # 注意这里k、q、v的维度实际上是[batch,seq_len,d_model]
        q=self.q_l(x_q)
        k=self.k_l(x_k)
        v=self.v_l(x_v)

        # 计算原始q的维度
        batch_size, seq_len, d_model = q.size()
        # 计算每个头的k、q、v的维度
        self.d_k=d_model//self.n_heads

        q = q.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k = k.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v = v.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        # 转换后变成[batch_size,n_heads,seq_len,d_k]
        # q = q.view(batch_size,seq_len,self.n_heads,self.d_k).transpose(1,2)
        # k = k.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        # v = v.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        # 计算attention
        scores=self.attention(q, k, v, self.d_k, mask, self.dropout)

        # 连接多个头的结果，并给出最后的输出
        batch_size, n_head, seq_len, d_k = scores.size()
        concat=scores.transpose(1, 2).contiguous().view(batch_size,seq_len,d_model) # contiguous()确保内存连续性
        output=self.out(concat)

        return output







