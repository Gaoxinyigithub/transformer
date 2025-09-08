import math

import torch
from torch import nn
from torch.nn import functional as F

class Norm(nn.Module):
    def __init__(self,d_model,eps=1e-6):
        super().__init__()
        self.size=d_model

        # 归一化层的两个可学习参数
        self.alpha=nn.Parameter(torch.ones(self.size))
        self.bias=nn.Parameter(torch.zeros(self.size))
        self.eps=eps

    def forward(self,x):
        norm=self.alpha*(x-x.mean(dim=-1,keepdim=True))/(x.std(dim=-1,keepdim=True)+self.eps)+self.bias #在分母中加self.eps是为了防止分母为0
        return norm


    