# transformer : attention is all you need
环境搭建：

`conda create -n transformer_env python=3.8`

激活环境：

`conda activate transformer_env`

进入环境：

`conda activate transformer_env`

下载相关的包：

`pip install -e .`

下载对应的语言包：

如果你的包的版本跟我的一样，我已经将需要的gz文件放在`transformer->model`下可以直接使用下面的命令安装。

`pip install transformer/model/en_core_web_sm-3.1.0.tar.gz`

`pip install transformer/model/zh_core_web_sm-3.1.0.tar.gz`

快速体验：(注：这里为了计算快epoch=2，可以通过增加epoch来优化训练效果)

`cd transformer`

`python train.py`

--------------------------

具体讲解见：

https://blog.csdn.net/m0_47719040/article/details/150608889?spm=1001.2014.3001.5502



https://blog.csdn.net/m0_47719040/article/details/151180196?sharetype=blogdetail&sharerId=151180196&sharerefer=PC&sharesource=m0_47719040&spm=1011.2480.3001.8118



部分内容如下

## 1 整体框架

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/ae8c2f7c867a4150b5cdf58e43f8bbf7.png#pic_center)
观察上面的图，实际上transformer本身的架构非常简单，整体上就是由“N个编码器+N个解码器+全连接层”组成。内部包含的一些重要模块将在下面逐一进行详细的分析。

## 2、位置编码

### 2.1 原理

位置编码公式：

$PE(pos,2i)=sin(\frac{pos}{10000^{2i/d\_model}})$

$PE(pos,2i+1)=cos⁡(\frac{pos}{10000^{2i/d\_model}})$

**那为什么要这样做位置编码呢？**

现在，考虑位置 $pos+k$ 的编码。根据三角函数的加法公式：

$sin(a+b)=sin(a)cos(b)+cos(a)sin(b)$

$cos⁡(a+b)=cos⁡(a)cos⁡(b)−sin⁡(a)sin⁡(b)cos(a+b)=cos(a)cos(b)−sin(a)sin(b)$

令 $a=\frac{pos}{10000^{2i/d\_model}}$​ 和 $b=\frac{k}{10000^{2i/d\_model}}$。则：

$PE(pos+k,2i)=sin⁡(a+b)=sin⁡(a)cos⁡(b)+cos⁡(a)sin⁡(b)$

$PE(pos+k,2i+1)=cos⁡(a+b)=cos⁡(a)cos⁡(b)−sin⁡(a)sin⁡(b)$

注意，实际上

$sin(a)=PE(pos,2i)$

$cos⁡(a)=PE(pos,2i+1)$

所以，有

$PE(pos+k,2i)=PE(pos,2i)⋅cos(b)+PE(pos,2i+1)⋅sin(b)$

$PE(pos+k,2i+1)=PE(pos,2i+1)⋅cos⁡(b)−PE(pos,2i)⋅sin⁡(b)$

这表明 $PE(pos+k)$的每个维度都可以表示为 $PE(pos)$ 的对应维度的线性组合。系数$cos(b)$ 和 $sin⁡(b)$只依赖于偏移量 $k$ 和维度 $i$，但对于固定的 $k$，这些系数是常数。

### 2.2 代码

```python
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
        super(PositionalEncoding).__init__()
        # 创建一个与输入矩阵相同大小的矩阵（用于与输入矩阵相加）
        # 形状为 [max_len, d_model]
        self.encoding = torch.zeros(max_len,d_model,device=device)
        self.encoding.requires_grad = False #由于位置编码是固定的,所以不需要进行梯度计算

        # 创建位置索引[0,1,2,3,````,max_len-1]
        pos = torch.arange(0, max_len, device=device)
        pos = pos.float().unsqueeze(dim=1)

        # 创建维度索引
        _2i=torch.arange(0,d_model,device=device)

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
        x=x+self.encoding[:seq_len,:]
        return x
```

## 3、多头注意力机制

实际上transformer架构中用到了两种多头注意力机制，一种是没有掩码的注意力机制，另一种是带掩码的注意力机制。掩码的作用实际上就是控制模型能看到多少东西，落到具体操作上实际上就是乘了一个矩阵。

### 3.1 原理

注意力实际上就是把重点放到某些重要的东西上，同时忽略一些东西。想像以下人在一个大环境中玩捉迷藏游戏，一定会有查找的先后顺序和重点关注位置，同样的对于注意力机制也是这样的，多头注意力机制可以理解为从不同的方面来找重点。比如有些人在环境中更注重建筑结构，有些则觉得颜色更重要，还有些对气味比较敏感。

这部分主要包含以下三块内容：

 - 多头注意力机制的原理
 - mask的实现
 - 多头的代码实现

1、多头注意力机制
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/640b7d9aefaa4e9b97698111da34e0cf.png#pic_center)


假设现在有一个句子：I am Lily

  1. 第一步：将单词编码成向量，即单词 I、 am、Lily三个单词分别编码成向量。
  2. 第二步：每个单词生成其对应的q(query)、k(key)、v(value)，实际上就是通过三个全连接层做了三个参数不同的线性变化。
  3. 第三步：基于每个单词生成的q(query)、k(key)、v(value)计算最后的输出，计算公式：$z=softmax(\frac{q*k^T}{\sqrt{d_k}})*v$，其中$d_k$是$q$的维度。
     $q*k^T$的部分如下，在这个案例中$d_k=2$，可以看到实际上生成的矩阵可以理解为单词间的相关性，矩阵中的第一行是第一个单词和所有单词的相关性，第二行是第二个单词和所有单词的相关性，依次类推，我们已知单词和单词之间并不是独立的，或者说上下文之间是有关系的，那么实际上注意力机制就在学习这种关系，并将这个关系融合在后面的输出特征里。
     ![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/fa780c9c55de43a9b634ff31155a48e1.jpeg#pic_center)忽略softmax我们来看这个式子的后半部分$softmax(\frac{q*k^T}{\sqrt{d_k}})*v$，很容易能够看出$z_1$实际上按之前的相关性集合了$v_1、v_2、v_3$的特征，所以通过attention可以将上下文信息融合，同时实际上读者如果了解主成分分析或发现它们的原理是类似的。
     ![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/2a2886d5ded7412b88e9402e93ad7574.jpeg#pic_center)

2、mask掩码(decoder的部分需要的)

mask掩码实际上是希望网络能看到我们想给它看的信息，不能看到我们不给它看的信息。想像一下我们自己在阅读的时候实际上是读到某一个位置才知道这个位置在干嘛，同时因为我们读过前面的文章我们也知道前面在干嘛。

前面说attention的部分实际上是将上下文信息融合到当前的单词中，形成新的特征向量，那么读到某一个位置才看到某一个向里也很好理解，通俗来说就是单词一不要融合单词二和三，单词二可以融合一，但是三不行。那观察下图并用公式来理解
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/4742a9243e8d44508c2247d41b86713a.jpeg#pic_center)
$z_1=q_1\cdot k_1\cdot v_1+0\cdot v_2+0\cdot v_3$
$z_2=q_2\cdot k_1\cdot v_1+q_2\cdot k_2 \cdot v_2+0\cdot v_3$
$z_3=q_3\cdot k_1\cdot v_1+q_3\cdot k_2 \cdot v_2+q_3\cdot k_3\cdot v_3$

通过这样的方式我们就能让$attention$看到它们能看的，忽略看不到的信息。实际在计算的过程中是在$\frac{q*k^T}{\sqrt{d_k}}$的基础上乘用一个mask掩码矩阵来判断哪里能看到哪里看不到，通过上面的内容我们很容易可以知道实际上mask矩阵就是一个0-1矩阵：
$\begin{bmatrix}
1 & 0 & 0\\
1 & 1 & 0\\
1 & 1& 1
\end{bmatrix}$

```python
scores=scores.masked_fill(mask==0,-1e9)
```

3、多头部分的实现

上面一直说的实际上是单头的部分，多头也很简单单头有一组$W_q、W_k、W_v$，那多头实际上就是有多组$W_q、W_k、W_v$，直观的看实际上就是下图这样
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/dca446392dfb4f47a6460bb52bacad55.jpeg#pic_center)在实际实现的过程中是一次性计算完成之后在分离$q_1^1$和$q_1^2$等，即如下图所示
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/db107922558d4b7986d8402fb56878bf.jpeg#pic_center)



### 3.2 代码

```python
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

        # 转换后变成[batch_size,n_heads,seq_len,d_k]
        q = q.view(batch_size,seq_len,self.n_heads,self.d_k).transpose(1,2)
        k = k.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        # 计算attention
        scores=self.attention(q, k, v, self.d_k, mask, self.dropout)

        # 连接多个头的结果，并给出最后的输出
        batch_size, n_head, seq_len, d_k = scores.size()
        concat=scores.transpose(1, 2).contiguous().view(batch_size,seq_len,d_model) # contiguous()确保内存连续性
        output=self.out(concat)

        return output
```

## 4、归一化层

### 4.1 原理

这里很简单$out=\alpha \cdot (x-mean(x))/\sqrt{\sigma^2+\epsilon}+\delta$
其中$\alpha$和$\delta$是可学习参数
$mean(x)是x的均值，\sigma是基于均值计算的方差。$

### 4.2 代码

```python
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
        norm=self.alpha*(x-x.mean(dim=-1,keepdim=True))/(x.std(dim=-1,keepdim=True)+self.eps)+self.bias 
        #在分母中加self.eps是为了防止分母为0
        return norm

```

## 5、前馈层

### 5.1 原理

该部分论文中已经给出公式，实现上也非常简单，前馈层主要是进行非线性变换和空间映射，从而增加模型的表达能力和复杂性。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/a21eeb9a083b4ab1afd35d0160f18be5.jpeg#pic_center)

### 5.2 代码

```python
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
```

## 6、编码层

### 6.1 原理

单独的编码器层实际上是由“多头注意力机制+前馈层+残差连接&归一化层”构成的。
整体的transformer中的编码部分是由“词嵌入层+位置编码层+n个EncoderLayer”构成。
具体见代码实现部分，只是将前面提到的块进行组合。

### 6.2 代码

```python
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

```

## 7、解码层

### 7.1 原理

单独的解码层，这个结构中包含“多头注意力机制+残差连接&归一化层+encoder-decoder注意力机制+残差连接&归一化层+前馈层+残差连接&归一化层”。
transformer解码器中包含“词嵌入层+位置编码层+n个DecoderLayer”。
具体见代码实现部分，只是将前面提到的块进行组合。

### 7.2 代码

```python
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
        ff_out=nn.Dropout(ff_out)
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
```

## 8、整体模型

### 8.1 原理

最后我们对整体代码做拼接即“N个编码器+N个解码器+全连接层”。
具体见代码实现部分，只是将前面提到的块进行组合。

### 8.2 代码

```python
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
```

