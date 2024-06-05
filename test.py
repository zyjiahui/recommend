import torch
# (2,3) - (batch_size,num_features)
x = torch.tensor([
                [1,2,3],
                [4,5,6]
                ])  
# (3,4) - (num_features,embedding_dim)
enbedding_vector = torch.tensor([
                                [0.11,0.33,0.55,0.63],
                                [0.44,0.34,0.56,0.88],
                                [0.56,0.78,0.99,0.45]
                                ])    
embedding_x = x.unsqueeze(2) * enbedding_vector.unsqueeze(0)   # 这里会发生广播
# (batch_size,num_features,1) * (1,num_features,embedding_dim) = (batch_size,num_features,embedding_dim)
print(embedding_x)


# 二维向量 在dim=1上求和，keepdim为True，保持结果为二维，但keepdim为False，结果降维成一维
tensor = torch.tensor([[1, 2, 3, 4],
                       [5, 6, 7, 8],
                       [9, 10, 11, 12]])
sum_result = tensor.sum(dim=1, keepdim=True)
sum_result1 = tensor.sum(dim=1, keepdim=False)
print(sum_result)
print(tensor.size())
print(sum_result.size())
print(sum_result1)
