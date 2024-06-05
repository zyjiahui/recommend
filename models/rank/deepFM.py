"""
DeepFM模型：
    在wide侧添加了FM因子分解机，用来捕捉二阶交叉特征，而不是wide&deep那种只有LR的一阶特征
"""
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim

# class DeepFM(nn.Module):
#     def __init__(self,num_features,embedding_dim,hidden_dim):
#         super(DeepFM,self).__init__()
#         self.linear = nn.Linear(num_features,1)
#         self.embedding = nn.Embedding(num_features,embedding_dim)
#         self.dnn = nn.Sequential(
#             nn.Linear(embedding_dim,hidden_dim),  # 
#             nn.ReLU(),
#             nn.Linear(hidden_dim,1)
#         )

#     def forward(self,x):
#         x = x.float()
#         embedding_vector = self.embedding(x)  # 把稀疏01特征转化为向量
#         # FM部分（一阶和二阶）
#         fm_one = torch.sum(self.linear(x),dim=1,keepdim=True)
#         fm_two = 0.5 * (torch.sum(embedding_vector,dim=1) ** 2 - torch.sum(embedding_vector ** 2,dim=1))
#         fm_output = fm_one + fm_two
#         # DNN部分
#         dnn_output = self.dnn(embedding_vector.view(-1,embedding_vector.size(1)))
#         # 两部分结合
#         output = torch.sigmoid(fm_output + dnn_output)
#         return output


# # 测试
# num_features = 10
# embedding_dim = 8
# hidden_dim = 32

# model = DeepFM(num_features,embedding_dim,hidden_dim)

# x = torch.randint(0,num_features,(1,num_features))

# output = model(x)
# print("结果：",output.item())




import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepFM(nn.Module):
    def __init__(self, num_features, embedding_dim, hidden_dim):
        super(DeepFM, self).__init__()
        self.num_features = num_features
        self.embedding_dim = embedding_dim
        
        # 线性部分
        self.linear = nn.Linear(num_features, 1)
        
        # 嵌入层
        self.embedding = nn.Embedding(num_features, embedding_dim)  #参数为特征数量和想要生成向量的维度，比如10个特征，生成embedding的维度为8，结果为（10，8）
        
        # DNN 部分
        self.dnn = nn.Sequential(
            nn.Linear(num_features * embedding_dim, hidden_dim),  # num_features * embedding_dim每一个输入进行了embedding之后的大小
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        # 确保输入 x 是 Float 类型
        x = x.float()
        
        # FM 部分（一阶部分）
        fm_one = self.linear(x)
        
        # 嵌入层，将每个特征索引映射到嵌入向量
        embedding_vector = self.embedding(torch.arange(self.num_features))  # 将 0-num_features的整数序列放入到embedding层生成对应的向量
        # (num_features, embedding_dim)
        
        # 将嵌入向量扩展到批量维度并进行点积计算   unsqueeze函数在指定的位置插入一个大小为1的新维度tensor.unsqueeze(dim)，dim为新维度的位置索引
        # 直接用 embedding 进行二阶交叉时，我们计算的是嵌入向量本身的交叉特征，而没有考虑输入特征的具体值。如果仅仅对嵌入向量进行操作，我们将忽略特征 x 的值在计算交叉特征时的影响。而扩展后的 embedded_x 引入了特征 x 的值，因此在计算交叉特征时考虑了每个特征值的具体影响。
        embedded_x = x.unsqueeze(2) * embedding_vector.unsqueeze(0) # 二阶特征交叉时，使用了embedding和原特征两种数据
        # (batch_size, num_features, embedding_dim)
        
        # FM 部分（二阶部分）
        sum_of_squares = torch.sum(embedded_x, dim=1) ** 2
        square_of_sum = torch.sum(embedded_x ** 2, dim=1)
        fm_two = 0.5 * (sum_of_squares - square_of_sum).sum(dim=1, keepdim=True)  # keepdim的默认值是False   用来保持加和后的维度
        # (batch_size, 1)
        
        # DNN 部分
        dnn_input = embedded_x.view(x.size(0), -1)  # 展平  (batch_size, num_features * embedding_dim)
        dnn_output = self.dnn(dnn_input)
        # (batch_size, 1)
        
        # 将 FM 部分和 DNN 部分的输出结合
        output = fm_one + fm_two + dnn_output
        
        return torch.sigmoid(output)

# 测试
num_features = 10
embedding_dim = 8
hidden_dim = 32

model = DeepFM(num_features, embedding_dim, hidden_dim)

# 输入样本数据
x = torch.randint(0, 2, (1, num_features))  # 生成一个样本，特征是0或1

output = model(x)
print("结果：", output.item())

