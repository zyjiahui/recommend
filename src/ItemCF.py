# 基于物品的协同过滤算法：该算法不利用物品的内容属性计算物品间的相似性，而是通过分析用户的历史行为数据来计算物品的相似度，这样表示：物品A和C间的相似度高是因为喜欢物品A的用户及可能喜欢物品C
# 思路：1、计算目标物品与其他物品的相似性  2、在目标用户历史数据中找出与目标物品最相似的n个物品  3、根据目标用户对相似物品的喜爱程度来计算其对目标物品的喜爱程度

import numpy as np
import pandas as pd
def loadData():
    items = {
        'A': {'Alice': 5.0, 'user1': 3.0, 'user2': 4.0, 'user3': 3.0, 'user4': 1.0},
        'B': {'Alice': 3.0, 'user1': 1.0, 'user2': 3.0, 'user3': 3.0, 'user4': 5.0},
        'C': {'Alice': 4.0, 'user1': 2.0, 'user2': 4.0, 'user3': 1.0, 'user4': 5.0},
        'D': {'Alice': 4.0, 'user1': 3.0, 'user2': 3.0, 'user3': 5.0, 'user4': 2.0},
        'E': {'user1': 3.0, 'user2': 5.0, 'user3': 4.0, 'user4': 1.0}
    }
    return items

# 物品间的相似性矩阵
item_data = loadData()
similarity_matrix = pd.DataFrame(
    np.identity(len(item_data)),
    index=item_data.keys(),
    columns=item_data.keys()
)
# print(similarity_matrix)

for i1,users1 in item_data.items():
    for i2,users2 in item_data.items():
        if i1 == i2:
            continue
        vec1,vec2 = [],[]
        for user,rating1 in users1.items():
            rating2 = users2.get(user,-1)
            if rating2 == -1:
                continue
            vec1.append(rating1)
            vec2.append(rating2)
        similarity_matrix[i1][i2] = np.corrcoef(vec1,vec2)[0][1]
print(similarity_matrix)


# 从Alice购买的物品中找出与E最相似的num个物品
target_user = 'Alice'
target_item = 'E'
num = 2
sim_items = []
sim_items_list = similarity_matrix[target_item].sort_values(ascending=False).index.tolist()
print(sim_items_list)
for item in sim_items_list:  # 找出在Alice购买记录中出现的
    if target_user in item_data[item]:
        sim_items.append(item)
    if len(sim_items) == num:
        break
print(f'在Alice购买记录中，与物品{target_item}最相似的{num}个物品为{sim_items}')


# 预测用户Alice对物品E的评分
target_item = 'E'
target_user_mean_rating = np.mean(list(item_data[target_item].values()))
weighted_scores = 0
corr_values_num = 0
for item in sim_items:
    corr_value = similarity_matrix[target_item][item]
    user_mean_rating = np.mean(list(item_data[item].values()))

    weighted_scores += corr_value * (item_data[item][target_user] - user_mean_rating)
    corr_values_num += corr_value
target_item_pre = target_user_mean_rating + weighted_scores/corr_values_num
print(f'用户{target_user}对物品{target_item}的预测评分为：{target_item_pre}')