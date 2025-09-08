import math

import torch
from torch import nn
from torch.nn import functional as F
from transformer.MultiHeadAttention import MultiHeadAttention
from transformer.FeedForward import FeedForward
from transformer.Normalize import Norm
from transformer.PositionalEncoding import PositionalEncoding
'''
DecoderLayer 是单独的解码层，这个结构中包含“多头注意力机制+残差连接&归一化层+encoder-decoder注意力机制+残差连接&归一化层+前馈层+残差连接&归一化层”
Decoder是transformer解码器中的部分，包含“词嵌入层+位置编码层+n个DecoderLayer”
'''

class DecoderLayer(nn.Module):
    '''
    DecoderLayer 是单独的解码层，这个结构中包含“多头注意力机制+残差连接&归一化层+encoder-decoder注意力机制+残差连接&归一化层+前馈层+残差连接&归一化层”
    根据上面的结构已知实际上单独一个解码器块包含以下部分
    1、两个多头注意力机制
    2、一个前馈层
    3、三个归一化层
    4、三个dropout层
    '''
    def __init__(self,d_model, n_heads, dropout=0.1):
        super().__init__()
        # 两个注意力层
        self.atten1=MultiHeadAttention(n_heads,d_model,dropout)
        self.atten2 = MultiHeadAttention(n_heads, d_model, dropout)
        # 一个前馈层
        self.ff=FeedForward(d_model,d_ff=2048,dropout=dropout)
        # 三个归一化层
        self.norm1 = Norm(d_model)
        self.norm2 = Norm(d_model)
        self.norm3 = Norm(d_model)
        # 三个dropout层
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self,x,e_outputs,mask1,mask2):
        # 第一个多头注意力机制
        attenout_1=self.atten1(x,x,x,mask1)
        # dropout层
        attenout_1=self.dropout1(attenout_1)
        # 残差连接层
        x=x+attenout_1
        # 归一化层
        x=self.norm1(x)

        # 第二个多头注意力机制
        attenout_2 = self.atten2(x, e_outputs, e_outputs, mask2)
        # dropout层
        attenout_2 = self.dropout2(attenout_2)
        # 残差连接层
        x = x + attenout_2
        # 归一化层
        x = self.norm2(x)

        # 前馈层
        ff_out=self.ff(x)
        # dropout层
        ff_out=self.dropout3(ff_out)
        # 残差连接层
        x = x + ff_out
        # 归一化层
        x = self.norm3(x)
        return x

class Decoder(nn.Module):
    def __init__(self,vocab_size,d_model,N_Decoderlayers,n_heads,dropout,device):
        super().__init__()
        self.N = N_Decoderlayers
        # 词编码
        self.embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
        # 位置编码
        self.pe = PositionalEncoding(d_model, max_len=80, device=device)
        # 多层解码器
        self.layers = nn.ModuleList([DecoderLayer(d_model=d_model,
                                                  n_heads=n_heads,
                                                  dropout=dropout)
                                     for _ in range(self.N)])

    def forward(self, x, e_outputs, mask1,mask2):
        # 词编码
        x = self.embed(x)
        # 位置编码
        x = self.pe(x)
        # 多层解码器
        for layer in self.layers:
            x = layer(x, e_outputs,mask1,mask2)
        return x
