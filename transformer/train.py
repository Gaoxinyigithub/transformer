
'''
所有模型训练实际上都包含以下几个步骤
1、读取训练需要的数据
2、将数据分为训练集和测试集
3、创建模型进行训练
'''

'''
训练模型需要的数据放在data文件夹下，该文件夹下包含一个english.txt和一个french.txt
首先读取该txt文件
'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from transformer.Transformer import Transformer
from transformer.batch import create_masks
from transformer.process import *
import numpy as np
import time

# 数据
src_file = '/home/gxy/PycharmProjects/reinforcement_learning/transformer/data/en.txt'
trg_file = '/home/gxy/PycharmProjects/reinforcement_learning/transformer/data/zh.txt'
src_lang = 'en_core_web_sm'
trg_lang = 'zh_core_web_sm'
max_strlen = 80
batchsize = 1500
src_data, trg_data = read_data(src_file, trg_file) # 读取文件内容
EN_TEXT, ZH_TEXT = create_fields(src_lang, trg_lang) # 设置分词（token）的标准
# 分词形成字典，并进行数据集做批处理
train_iter, src_pad, trg_pad = create_dataset(src_data, trg_data, EN_TEXT, ZH_TEXT, max_strlen, batchsize)

'''模型训练'''
# 模型参数定义
d_model = 512
heads = 8
N = 6
dropout = 0.1
src_vocab = len(EN_TEXT.vocab) # 英文的数据集中的token的个数
trg_vocab = len(ZH_TEXT.vocab) # 中文的数据集中的token的个数
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu' # 由于torchtext的版本需要的比较低，而我的cuda版本比较高，所以torch无法使用cuda如果你可以可以将这一行注释掉，并取消上一行的注释
model = Transformer(src_vocab, trg_vocab, d_model, N, heads, dropout,device=device)
optim = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

for p in model.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)



# 模型训练
def train_model(epochs, print_every=10):
    model.train() # 将模型设置为训练模式

    start = time.time() # 记录训练开始时间
    temp = start # 临时时间变量，用于计算间隔时间

    total_loss = 0 #累积损失

    for epoch in range(epochs): # 外层循环：遍历所有训练轮次
        for i, batch in enumerate(train_iter): # 内层循环：遍历训练数据中的所有批次
            src = batch.src.transpose(0, 1) # 转置源语言序列维度
            trg = batch.trg.transpose(0, 1) # 转置目标语言序列维度

            trg_input = trg[:, :-1] # 目标序列输入（去掉最后一个token）

            # 记录目标，方便后续求loss function
            targets = trg[:, 1:].contiguous().view(-1) # 目标序列标签（去掉第一个token）并展平


            # 使用掩码代码创建函数来制作掩码
            src_mask, trg_mask = create_masks(src, trg_input, src_pad, trg_pad)
            preds = model(src, trg_input, src_mask, trg_mask) # 向前传播预测结果

            optim.zero_grad() # 清零梯度

            loss = F.cross_entropy(preds.view(-1, preds.size(-1)),
                                   targets, ignore_index=trg_pad) # 计算交叉熵损失
            loss.backward() # 反向传播计算梯度
            optim.step() # 跟新模型参数

            total_loss += loss.item() # 累积损失
            if (i + 1) % print_every == 0: # 每隔一定迭代次数打印进度
                loss_avg = total_loss / print_every
                print("time = %dm, epoch %d, iter = %d, loss = %.3f, %ds per %d iters" %
                      ((time.time() - start) // 60, epoch + 1, i + 1, loss_avg,
                       time.time() - temp, print_every))
                total_loss = 0
                temp = time.time()
        # 训练结束后保存最终模型
        final_checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optim.state_dict(),
            'epoch': epochs,
            'ZH_TEXT': ZH_TEXT,
            'EN_TEXT': EN_TEXT,
            'src_pad':src_pad,
            'trg_pad':trg_pad
        }
        torch.save(final_checkpoint, 'model_final.pth')
        print("最终模型已保存为: model_final.pth")



train_model(50)

