#!python
# -*- coding: utf-8 -*-
# @author: Kun

'''
Author: Kun
Date: 2021-12-16 10:54:00
LastEditTime: 2022-05-16 14:40:37
LastEditors: Kun
Description: 
FilePath: /my_open_projects/NLP4CyberSecurity/embedding/word2vec.py
'''


import os
import csv
import pickle
import time
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from utils.seg_util import GeneSeg

from gensim.models.word2vec import Word2Vec

CACHE_DIR  = "./cache/xss_injection"

learning_rate = 0.1  # 学习率
# 单词库  3000，用来设置数据的 10,000 个单词。这是我们输入的独热向量，在向量中仅有一个值为「1」的元素是当前的输入词，其他值都为「0」
vocabulary_size = 3000
batch_size = 128  # batch的大小
embedding_size = 128  # 单词转为稠密向量的维度，这里使用128作为词向量的维度，隐藏层的大小,默认为100
num_skips = 4  # 为对每个单词生成多少个样本，不能大于skip_windows的两倍，并且batch_size是其整数倍
skip_window = 5  # 指单词最远可以联系的距离，设为1表示只能跟紧邻的两个单词生成样本。默认为5.
num_sampled = 64  # 训练时用作负样本的噪声单词的数量
num_iter = 5
plot_only = 100
log_dir = "word2vec.log"

plt_dir = os.path.join(CACHE_DIR, "word2vec.png")
vec_dir = os.path.join(CACHE_DIR, "word2vec.pickle")


################################################################################################
# 构建数据集
def build_dataset(datas, words):
    count = [["UNK", -1]]  # list  此时，len(count)=1,表示只有一组数据  词汇表vocabulary
    # print("count"+str(len(count)))
    counter = Counter(words)
    count.extend(counter.most_common(vocabulary_size-1))
    # 将words中，最常见的49999个单词和对应的个数，放入count中。
    # 此时，len(count)=50000,表示只有50000组数据
    # 使用collections.Counter统计word单词列表中单词的频数，然后使用most_common方法取
    # top 50000频数的单词作为词汇表vocabulary
    # 使用了most_common,所以count中的word，是按照word在文本中出现的次数从大到小排列的
    vocabulary = [c[0] for c in count]
    data_set = []
    for data in datas:
        d_set = []
        for word in data:
            if word in vocabulary:
                d_set.append(word)
            else:
                d_set.append("UNK")
                count[0][1] += 1
        data_set.append(d_set)
    return data_set

################################################################################################


def plot_with_labels(low_dim_embs, labels, filename=plt_dir):
    plt.figure(figsize=(10, 10))  # in inches
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y)  # 展示单词本身
        plt.annotate(label, xy=(x, y), xytext=(5, 2),
                     textcoords="offset points",
                     ha="right",
                     va="bottom")
        f_text = "vocabulary_size=%d;batch_size=%d;embedding_size=%d;skip_window=%d;num_iter=%d" % (
            vocabulary_size, batch_size, embedding_size, skip_window, num_iter
        )
        plt.figtext(0.03, 0.03, f_text, color="green", fontsize=10)
    plt.show()
    plt.savefig(filename)  # 保存文件

################################################################################################


def save(embeddings):
    dictionary = dict([(embeddings.index2word[i], i)
                       for i in range(len(embeddings.index2word))])
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    word2vec = {"dictionary": dictionary, "embeddings": embeddings,
                "reverse_dictionary": reverse_dictionary}
    with open(vec_dir, "wb") as f:
        pickle.dump(word2vec, f)

################################################################################################


def make_vec():

    start = time.time()
    words = []
    datas = []
    with open("data/xssedtiny.csv", "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, fieldnames=["payload"])
        for row in reader:
            payload = row["payload"]
            word = GeneSeg(payload)
            datas.append(word)
            words += word

    data_set = build_dataset(datas, words)

    model = Word2Vec(data_set, size=embedding_size,
                     window=skip_window, negative=num_sampled, iter=num_iter)
    embeddings = model.wv

    # tsne 降维可视化

    # TSNE实现降维
    tsne = TSNE(perplexity=30, n_components=2, init="pca", n_iter=5000)
    # 直接将原始的128维的嵌入向量降到2维，再用前面的plot_with_labels进行显示
    plot_words = embeddings.index2word[:plot_only]
    plot_embeddings = []
    for word in plot_words:
        plot_embeddings.append(embeddings[word])
    low_dim_embs = tsne.fit_transform(plot_embeddings)
    plot_with_labels(low_dim_embs, plot_words)

    save(embeddings)
    end = time.time()
    print("Over word2vec job in ", end-start)
    print("Saved words vec to", vec_dir)


if __name__ == "__main__":
    make_vec()
