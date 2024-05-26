# 基于用户的协同过滤算法  皮尔逊相关性系数

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

# 一、初始化用户相似性矩阵   训练数据中包含5个用户，这里用户的相似性矩阵维度也是 5 X 5
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

# print(user_data.items())
# 二、遍历每条用户-物品评分数据，算出的皮尔逊相似性系数来更新相似性矩阵
for u1,items1 in user_data.items():  # 遍历用户和对用的物品评分
    for u2,items2 in user_data.items():
        # print(u2,items2)
        if u1 == u2:  # 内层循环遍历的是外层循环的内容，同一个用户，直接跳入下一次循环，保留原单位矩阵的1
            continue
        vec1,vec2 = [],[]  # 存储u1和u2对同一个物品的评分
        for item,rating1 in items1.items():  # 遍历当前物品的每一个评分
            # print(item,rating1)
            rating2 = items2.get(item,-1)
            # print(rating2)
            if rating2 == -1:
                continue
            vec1.append(rating1)
            vec2.append(rating2)
        # 计算不同用户之间的皮尔逊相关系数
        similarity_matrix[u1][u2] = np.corrcoef(vec1,vec2)[0][1]
print(similarity_matrix)

# 三、计算与Alice最相似的num个用户
target_user = 'Alice'
num = 2
# 最相似的用户是自己，所以去除本身
# sort_values(ascending=False) 降序排序  [1:num+1]切片提取1:num+1个，  .index.tolist() 将取出的索引转化为列表形式
sim_users = similarity_matrix[target_user].sort_values(ascending=False)[1:num+1].index.tolist()  
print(f'与用户{target_user}最相似的{num}个用户为：{sim_users}')


# 四、预测Alice对物品E的评分   
# 基于与Alice最相似的num个用户（这里用皮尔逊相关性系数算出）对物品E的评分来计算Alice对物品E的评分
weighted_scores = 0  # 计算皮尔逊的中间变量 分子：   目标用户与其相似用户的相似度 * （相似用户对物品E的评分 - 相似用户对所有物品历史评分的均值）  的累加
corr_values_num = 0  # 计算皮尔逊的中间变量 分母：   目标用户与相似用户的相似度 的累加
target_item = 'E'  # 目标物品  求Alice对E的评分
for user in sim_users:
    corr_value = similarity_matrix[target_user][user]  # 相似性矩阵中目标用户Alice和sim_users们的相似性
    user_mean_rating = np.mean(list(user_data[user].values()))  # 对相似性用户们的物品评分求均值

    weighted_scores += corr_value * (user_data[user][target_item] - user_mean_rating)
    corr_values_num += corr_value

target_user_mean_rating = np.mean(list(user_data[target_user].values()))  # 目标用户对物品历史评分的均值
# 计算皮尔逊 公式
target_user_pre = target_user_mean_rating + weighted_scores/corr_values_num
print(f'用户{target_user}对物品{target_item}的预测评分为：{target_user_pre}')
