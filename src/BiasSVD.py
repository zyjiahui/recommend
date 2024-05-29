import random
import math

class BiasSVD():
    def __init__(self,rating_data,F=5,alpha=0.1,lmbda=0.1,max_iter=100):
        self.F = F  # 隐向量的维度 
        self.P = dict()  # 用户矩阵 P [users_num,F]
        self.Q = dict()  # 物品矩阵 Q [items_num,F]
        self.bu = dict()  # 用户偏置系数
        self.bi = dict()  # 物品偏置系数
        self.mu = 0  # 全局偏置参数
        self.alpha = alpha  # 学习率
        self.lmbda = lmbda  # 正则项系数
        self.max_iter = max_iter  # 最大迭代次数
        self.rating_data = rating_data  # 评分矩阵

        for user,items in self.rating_data.items():
            self.P[user] = [random.random() / math.sqrt(self.F) for x in range(0,F)]  # 生成长度为F的列表，隐向量矩阵的维度
            self.bu[user] = 0
            for item,rating in items.items():
                if item not in self.Q:  # 确保每一个物品只初始化一次     'user1': {'item1': 5, 'item2': 3},'user2': {'item1': 4, 'item3': 2}  其中Q[item1]不用在两次循环(user1和user2两次循环)中都初始化
                    self.Q[item] = [random.random() / math.sqrt(self.F) for x in range(0,F)]
                    self.bi[item] = 0


    # 采用梯度下降的方式训练模型参数
    def train(self):
        cnt,mu_sum = 0,0  # 计数，评分总和
        for user,items in self.rating_data.items():
            for item,rui in items.items():
                mu_sum,cnt = mu_sum + rui,cnt + 1  # 分数累加，计数+1
        self.mu = mu_sum / cnt  # 计算mu，全局平均

        for step in range(self.max_iter):
            # 遍历所有用户和历史交互物品
            for user,items in self.rating_data.items():
                # 遍历历史交互物品
                for item,rui in items.items():
                    rhat_ui = self.predict(user,item)  # 评分预测
                    e_ui = rui - rhat_ui  # 评分预测偏差

                    # 参数更新
                    self.bu[user] += self.alpha * (e_ui - self.lmbda * self.bu[user])
                    self.bi[item] += self.alpha * (e_ui - self.lmbda * self.bi[item])
                    for k in range(0,self.F):  # 隐向量0 - F-1
                        self.P[user][k] += self.alpha * (e_ui * self.Q[item][k] - self.lmbda * self.P[user][k])
                        self.Q[item][k] += self.alpha * (e_ui * self.P[user][k] - self.lmbda * self.Q[item][k])
            self.alpha *= 0.1
            


    # 评分预测
    def predict(self,user,item):
       return sum(self.P[user][f] * self.Q[item][f] for f in range(0,self.F)) + self.bu[user] + self.bi[item] + self.mu

def loadData():
    rating_data={1: {'A': 5, 'B': 3, 'C': 4, 'D': 4},
            2: {'A': 3, 'B': 1, 'C': 2, 'D': 3, 'E': 3},
            3: {'A': 4, 'B': 3, 'C': 4, 'D': 3, 'E': 5},
            4: {'A': 3, 'B': 3, 'C': 1, 'D': 5, 'E': 4},
            5: {'A': 1, 'B': 5, 'C': 5, 'D': 2, 'E': 1}
            } 
    return rating_data

rating_data = loadData()
basicsvd = BiasSVD(rating_data,F=10)
basicsvd.train()
# 预测用户1对物品E的评分
for item in ['E']:
    print(item,basicsvd.predict(1,item))