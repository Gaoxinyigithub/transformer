import math

import torch
from torch import nn
from torch.nn import functional as F
from transformer.MultiHeadAttention import MultiHeadAttention
from transformer.FeedForward import FeedForward
from transformer.Normalize import Norm
from transformer.PositionalEncoding import PositionalEncoding


'''
EncoderLayer 是单独的编码层，这个结构中包含“多头注意力机制+前馈层+残差连接&归一化层”
Encoder是transformer编码器中的部分，包含“词嵌入层+位置编码层+n个EncoderLayer”
'''
class EncoderLayer(nn.Module):
    def __init__(self,d_model,n_heads,dropout=0.1):
        super().__init__()
        ## 多头注意力机制
        self.attention=MultiHeadAttention(n_heads, d_model,dropout=0.1)

        ## 前馈层
        self.ff=FeedForward(d_model,dropout=dropout)

        ## 归一化和dropout
        self.norm1=Norm(d_model)
        self.norm2=Norm(d_model)
        self.dropout1=nn.Dropout(dropout)
        self.dropout2=nn.Dropout(dropout)

    def forward(self,x,mask):
        # 多头注意力机制
        attn_output=self.attention(x,x,x,mask)
        # dropout层
        attn_output=self.dropout1(attn_output)
        # 残差连接层
        x=x+attn_output
        # 归一化层
        x=self.norm1(x)
        # 前馈层
        ff_out=self.ff(x)
        # dropout层
        ff_out=self.dropout2(ff_out)
        # 残差连接层
        x=x+ff_out
        # 归一化层
        x=self.norm2(x)
        return x

'''
nn.Embedding详解
nn.Embedding是PyTorch中的一个常用模块，其主要作用是将输入的整数序列转换为密集向量表示。在自然语言处理（NLP）任务中，可以将每个单词表示成一个向量，从而方便进行下一步的计算和处理。
torch.nn.Embedding(num_embeddings, 
                   embedding_dim, 
                   padding_idx=None, 
                   max_norm=None, 
                   norm_type=2.0, 
                   scale_grad_by_freq=False, 
                   sparse=False, 
                   _weight=None, 
                   _freeze=False, 
                   device=None, 
                   dtype=None)
必需参数：
num_embeddings (int)： 嵌入字典的大小，即词汇表的大小
embedding_dim (int)：每个嵌入向量的维度
可选参数：
padding_idx (int, 可选)：填充符号的索引
max_norm (float, 可选)：最大范数值
norm_type (float, 可选)：计算范数时使用的p-范数类型
scale_grad_by_freq (bool, 可选)：是否根据频率缩放梯度

'''

'''
Encoder中包含“词编码+位置编码+n个编码块”
'''
class Encoder(nn.Module):
    def __init__(self,vocab_size,d_model,N_Encoderlayers,n_heads,dropout,device):
        super().__init__()
        self.N=N_Encoderlayers
        # 词编码
        self.embed=nn.Embedding(num_embeddings=vocab_size,embedding_dim=d_model)
        # 位置编码
        self.pe=PositionalEncoding(d_model, max_len=80, device=device)
        # 多层编码器
        self.layers=nn.ModuleList([EncoderLayer(d_model=d_model,
                                                  n_heads=n_heads,
                                                  dropout=dropout)
                                     for _ in range(self.N)])

    def forward(self,x,mask):
        # 词编码
        x=self.embed(x)
        # 位置编码
        x=self.pe(x)
        #多层编码器
        for layer in self.layers:
            x=layer(x,mask)
        return x














