import torch
from torch import nn

class PositionalEncoding(nn.Module):
    "计算位置编码"
    def __init__(self, d_model, max_len, device):
        """
        正弦位置编码类的构造函数

        :param d_model: 模型的维度（嵌入向量维度）
        :param max_len: 最大序列长度
        :param device: 硬件设备设置（CPU或GPU）
        """
        super().__init__()
        # 创建一个与输入矩阵相同大小的矩阵（用于与输入矩阵相加）
        # 形状为 [max_len, d_model]
        self.encoding = torch.zeros(max_len,d_model)
        self.encoding.requires_grad = False #由于位置编码是固定的,所以不需要进行梯度计算

        # 创建位置索引[0,1,2,3,````,max_len-1]
        pos = torch.arange(0, max_len)
        pos = pos.float().unsqueeze(dim=1)

        # 创建维度索引
        _2i=torch.arange(0,d_model,step=2).float()

        # 对于偶数编码索引(0,2,4,...)：使用正弦函数
        self.encoding[:,0::2]=torch.sin(pos/(10000**(_2i/d_model)))
        # 对于奇数编码索引(1,3,5,...)：使用余弦函数
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))
        # 正余弦函数的范围是[-1,1],用该词编码矩阵与原矩阵想加不会导致新的矩阵与原矩阵相差过远
        # 根据三角函数周期性(2pi一个周期),pos+k是pos的线性组合
    def forward(self,x):
        # 在使用这个函数之前，实际上会做初始化
        # 初始化过程中会有最大的seq和d_model
        # [max_len=512,d_model=512]
        batch_size,seq_len,_=x.size()
        # [batch_size=128,seq_len=30]
        x = x+self.encoding[seq_len,:]
        return x
    
### 得到的是位置编码的向量
