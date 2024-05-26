# 基于用户的协同过滤算法

import numpy as np
import pandas as pd

# 创建数据（用户-物品交互表）
def loadData():  
    # 用字典建立表（因为现实场景中，用户对物品的评分比较稀疏，用矩阵存储会存在大量的空缺值）
    users = {
        'Alice': {'A': 5, 'B': 3, 'C': 4, 'D': 4},
        'user1': {'A': 3, 'B': 1, 'C': 2, 'D': 3, 'E': 3},
        'user2': {'A': 4, 'B': 3, 'C': 4, 'D': 3, 'E': 5},
        'user3': {'A': 3, 'B': 3, 'C': 1, 'D': 5, 'E': 4},
        'user4': {'A': 1, 'B': 5, 'C': 5, 'D': 2, 'E': 1}
    }
    return users

# 计算用户相似性矩阵   训练数据中包含5个用户，这里用户的相似性矩阵维度也是 5 X 5
user_data = loadData()
print(len(user_data))  # 5
similarity_matrix = pd.DataFrame(  # pd.DataFrame创建一个二维数据结构
    np.identity(len(user_data)),  # 这里创建一个5*5的单位矩阵 （对角1，其余0）
    index=user_data.keys(),  # DataFrame的行索引
    columns=user_data.keys()  # DataFrame的列索引
)  # 这个相似性矩阵初始是一个单位矩阵（意味着用户只和自己完全相似，为1，与其他用户不相似，为0）
print(similarity_matrix)
#        Alice  user1  user2  user3  user4
# Alice    1.0    0.0    0.0    0.0    0.0
# user1    0.0    1.0    0.0    0.0    0.0
# user2    0.0    0.0    1.0    0.0    0.0
# user3    0.0    0.0    0.0    1.0    0.0
# user4    0.0    0.0    0.0    0.0    1.0
