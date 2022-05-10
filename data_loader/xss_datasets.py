#!python
# -*- coding: utf-8 -*-
# @author: Kun

'''
Author: Kun
Date: 2022-05-10 14:10:25
LastEditTime: 2022-05-10 14:10:40
LastEditors: Kun
Description: 
FilePath: /my_open_projects/NLP4CyberSecurity/data_loader/xss_datasets.py
'''


import os
import csv
import pickle
import random
import json
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import tensorflow as tf


CACHE_DIR  = "./cache/xss_injection"

vec_dir = os.path.join(CACHE_DIR, "word2vec.pickle")
pre_datas_train = os.path.join(CACHE_DIR, "pre_datas_train.csv")
pre_datas_test = os.path.join(CACHE_DIR, "pre_datas_test.csv")
process_datas_dir = os.path.join(CACHE_DIR, "process_datas.pickle")


def data_generator(data_dir):
    reader = tf.compat.v1.TextLineReader()
    queue = tf.compat.v1.train.string_input_producer([data_dir])
    _, value = reader.read(queue)
    # Start populating the filename queue.
    coord = tf.compat.v1.train.Coordinator()  # 创建一个协调器，管理线程
    sess = tf.compat.v1.Session()
    threads = tf.compat.v1.train.start_queue_runners(
        sess=sess, coord=coord)  # 启动QueueRunner, 此时文件名队列已经进队。
    while True:
        v = sess.run(value)
        [data, label] = v.split(b"|")
        data = np.array(json.loads(data.decode("utf-8")))
        label = np.array(json.loads(label.decode("utf-8")))
        yield (data, label)
    coord.request_stop()
    coord.join(threads)
    sess.close()


def batch_generator(datas_dir, datas_size, batch_size, embeddings, reverse_dictionary, train=True):
    batch_data = []
    batch_label = []
    generator = data_generator(datas_dir)
    n = 0
    while True:
        for i in range(batch_size):
            data, label = next(generator)
            data_embed = []
            for d in data:
                if d != -1:
                    data_embed.append(embeddings[reverse_dictionary[d]])
                else:
                    data_embed.append([0.0] * len(embeddings["UNK"]))
            batch_data.append(data_embed)
            batch_label.append(label)
            n += 1
            if not train and n == datas_size:
                break
        if not train and n == datas_size:
            yield (np.array(batch_data), np.array(batch_label))
            break
        else:
            yield (np.array(batch_data), np.array(batch_label))
            batch_data = []
            batch_label = []


def build_dataset(batch_size):
    with open(vec_dir, "rb") as f:
        word2vec = pickle.load(f)
    print(word2vec)
    embeddings = word2vec["embeddings"]
    reverse_dictionary = word2vec["reverse_dictionary"]
    train_size = word2vec["train_size"]
    test_size = word2vec["test_size"]
    dims_num = word2vec["dims_num"]
    input_num = word2vec["input_num"]
    train_generator = batch_generator(
        pre_datas_train, train_size, batch_size, embeddings, reverse_dictionary)
    test_generator = batch_generator(
        pre_datas_test, test_size, batch_size, embeddings, reverse_dictionary, train=False)
    return train_generator, test_generator, train_size, test_size, input_num, dims_num


if __name__ == "__main__":
    build_dataset()
