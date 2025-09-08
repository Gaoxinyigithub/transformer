import math

import torch
from torch import nn
from torch.nn import functional as F

class FeedForward(nn.Module):
    def __init__(self,d_model,d_ff=2048,dropout=0.1):
        super().__init__()
        # 实验结果表明，增大前馈层隐状态的维度有利于提高翻译结果的质量，实际上就是d_ff的值。
        self.linear_1=nn.Linear(d_model,d_ff)
        self.dropout=nn.Dropout(dropout)
        self.linear_2=nn.Linear(d_ff,d_model)

    def forward(self,x):
        x=self.dropout(F.relu(self.linear_1(x)))
        x=self.linear_2(x)
        return x
    