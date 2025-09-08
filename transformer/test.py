import torch
import torch.nn.functional as F
from torch.autograd import Variable
from transformer.Transformer import Transformer
from transformer.process import *
import numpy as np

# 加载保存的检查点
checkpoint_path = 'transformer/model_final.pth'
checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))  # 使用CPU加载，或指定GPU

# 恢复词汇表
ZH_TEXT = checkpoint['ZH_TEXT']
EN_TEXT = checkpoint['EN_TEXT']
# 恢复部分参数
src_pad=checkpoint['src_pad']
trg_pad=checkpoint['trg_pad']
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

# 恢复模型参数
model.load_state_dict(checkpoint['model_state_dict'])
# 恢复优化器状态 继续训练的时候才用的上
# optim = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
# optim.load_state_dict(checkpoint['optimizer_state_dict'])
# 获取训练轮次信息
trained_epochs = checkpoint['epoch']

print(f"模型已从 {checkpoint_path} 加载")
print(f"模型已训练 {trained_epochs} 个轮次")

'''
基于编码器-解码器架构的神经机器翻译模型的推理过程,用于将源语言句子翻译成目标语言。
在该代码中想要确定模型的输入和输出
'''
def translate(src, max_len=80, custom_string=False):
    """
    用于推理
    :param src: 输入源，可以是与处理好的张量或原始字符串
    :param max_len: 生成翻译的最大长度限制
    :param custom_string: 标志位，只是输入是否为原始字符串
    :return:
    """
    model.eval() # 设置为推理模式
    if custom_string == True: # 输入是否为原始字符
        src = tokenize_en(src, EN_TEXT) # 从Let me see. 变成向量 [89,21,95,2]
        src = torch.LongTensor(src) # 从向量 [89,21,95,2] 变成torch张量 tensor([89, 21, 95,  2])

    src_mask = (src != src_pad).unsqueeze(-2) # 张量中都不是1，新的张量tensor([[True, True, True, True]])
    e_outputs = model.encoder(src.unsqueeze(0), src_mask) # 计算编码器的输出
    # 4个数分别展开成512的向量，而且实际上已经是融合上下文信息的向量结果。
    outputs = torch.zeros(max_len).type_as(src.data) # 这里实际上相当于初始化一个输出
    outputs[0] = torch.LongTensor([ZH_TEXT.vocab.stoi['<sos>']])
    # ZH_TEXT.vocab.stoi是一个字典，是字符和数字的对应为了形成向量
    # <unk>：未知单词（out-of-vocabulary words）
    # <pad>：填充标记（用于使序列长度一致）
    # <sos>：序列开始标记 <eos>：序列结束标记
    for i in range(1, max_len): # 循环从1开始到max_len-1，i表示当前已生成的序列长度
        trg_mask = np.triu(np.ones((1, i, i)).astype('uint8'))
        trg_mask = Variable(torch.from_numpy(trg_mask) == 0)

        out = model.out(model.decoder(outputs[:i].unsqueeze(0), # 依次生成下一个输出
                                      e_outputs, trg_mask, src_mask))
        out = F.softmax(out, dim=-1) # 转为概率
        val, ix = out[:, -1].data.topk(1) # 获取概率和id

        outputs[i] = ix[0][0]
        if ix[0][0] == ZH_TEXT.vocab.stoi['<eos>']:
            break
    return ' '.join(
        [ZH_TEXT.vocab.itos[ix] for ix in outputs[:i]] # 最终结果
    )

words = 'Let me see.'
a=translate(words, custom_string=True)
print(a)