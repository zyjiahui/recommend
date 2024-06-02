"""
word2vec
one-hot存在很多缺点：忽略了词之间的相似性（每个词映射到高维空间中是相互正交的），编码矩阵过于稀疏

word2vec假设：文本中离得越近的词语相似度越高
    1、算法手段：CBOW 和 skip-gram来计算词向量矩阵  
    CBOW：根据上下文词预测中心词  （连续词袋）
    多一个平均池化操作
    skip-gram：根据中心词预测上下文词  （跳元模型）（常用这个方法）
    根据一个中心词，如何确定哪些词可以作为上下文词语，设置一个窗口，指定上下文包含哪些词语
    2、优化方法：负例采样（常用），层序softmax
    负采样：
    为什么要负采样？？？ skip-gram和CBOW都属于softmax多分类预测，且类别数目是整个词典的大小。负例指的是不与中心词c同时出现在窗口的词
    负采样是为了优化这一计算开销，因为softmax计算某一个类的概率时，分母是整个词典的大小，计算量巨大



word2vec缺点：
    没有考虑多义词
    窗口长度有限
    没有考虑全局的文本信息
    不是严格意义的语序


"""

from gensim.models import word2vec

if __name__ == '__main__':
    s1 = [0,1,2,3,4]
    s2 = [0,2,4,5,6]
    s3 = [2,3,4,4,6]
    s4 = [1,3,5,0,3]
    seqs = [s1,s2,s3,s4]
    model = word2vec.Word2Vec(seqs, vector_size=16, min_count=1)

    print(model.wv[1])

    print(model.wv.most_similar(1, topn=3))