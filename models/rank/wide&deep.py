"""
wide&deep模型：（兼顾记忆与扩展）
    wide部分：强记忆，用LR，一阶特征
    deep部分：Embedding扩展特征 + DNN深度神经网络（全连接的前馈网络）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class WideAndDeepModel(nn.Module):
    def __init__(self,wide_dim,deep_dim,hidden_dim):
        super(WideAndDeepModel,self).__init__()
        self.wide = nn.Linear(wide_dim,1)   # 这里输出维度是1，说明wide的线性层输出的是一个单一预测值
        self.deep = nn.Sequential(
            nn.Linear(deep_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,1)  # 这里输出维度也是1，deep层输出的也是单一预测值，  后面两个单一预测值相加，得出最后的预测值
        )

    def forward(self,x_wide,x_deep):
        wide_out = self.wide(x_wide)
        deep_out = self.deep(x_deep)
        output = torch.sigmoid(wide_out + deep_out)
        return output


wide_dim = 10
deep_dim = 20
hidden_dim = 32

model = WideAndDeepModel(wide_dim,deep_dim,hidden_dim)

x_wide = torch.randn(1,wide_dim)    # 这是输入到wide侧得特征，一般代表线性特征，这些特征与输出目标之间有直接的、可解释的关系
x_deep = torch.randn(1,deep_dim)    # 这是输入到deep侧得特征，一般为非线性的、复杂的特征，这些特征与输出目标之间的关系可解释性和直接关系较弱

output = model(x_wide,x_deep)
print("结果：",output.item())