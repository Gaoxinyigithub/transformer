import math

import torch
from torch import nn
from torch.nn import functional as F
from transformer.MultiHeadAttention import MultiHeadAttention
from transformer.FeedForward import FeedForward
from transformer.Normalize import Norm
from transformer.PositionalEncoding import PositionalEncoding
from transformer.Encode import Encoder
from transformer.Decode import Decoder

'''
整体transformer的框架实际上就是“N个编码器+N个解码器+全连接层”
'''

class Transformer(nn.Module):
    def __init__(self,src_vocab,trg_vocab,d_model,N,n_heads,dropout,device):
        super().__init__()
        # 编码器
        self.encoder=Encoder(src_vocab,d_model,N,n_heads,dropout,device)
        # 解码器
        self.decoder=Decoder(trg_vocab,d_model,N,n_heads,dropout,device)
        # 全连接层
        self.out=nn.Linear(d_model,trg_vocab)

    def forward(self,src,trg,src_mask,trg_mask):
        e_outputs=self.encoder(src,src_mask)
        d_outputs=self.decoder(trg,e_outputs,trg_mask,src_mask)
        output=self.out(d_outputs)
        return output