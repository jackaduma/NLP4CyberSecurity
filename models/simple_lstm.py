#!python
# -*- coding: utf-8 -*-
# @author: Kun


'''
Author: Kun
Date: 2021-12-16 11:22:09
LastEditTime: 2022-05-16 19:58:51
LastEditors: Kun
Description: 
FilePath: /my_open_projects/NLP4CyberSecurity/models/simple_lstm.py
'''


import os
import time
import numpy as np

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, InputLayer, Dropout, LSTM
from keras.callbacks import EarlyStopping, TensorBoard
from keras.optimizers import Adam
from keras.models import load_model
from sklearn.metrics import precision_score, recall_score


from data_loader.xss_datasets import build_dataset
from utils import init_session

CACHE_DIR = "./cache/xss_injection"

init_session()
batch_size = 8  # 128
epochs_num = 50  # 2
process_datas_dir = os.path.join(CACHE_DIR, "process_datas.pickle")

log_dir = "log/simple_lstm.log"
model_dir = os.path.join(CACHE_DIR, "simple_lstm_model")
# train_size=15381

tf.compat.v1.disable_eager_execution()


def train(train_generator, train_size, input_num, dims_num):
    print("Start Train Job! ")
    start = time.time()
    # inputs = InputLayer(input_shape=(
    #     input_num, dims_num.shape[0]), batch_size=batch_size)
    inputs = InputLayer(input_shape=(
        input_num, dims_num), batch_size=batch_size)
    layer1 = LSTM(128)
    output = Dense(2, activation="softmax", name="Output")
    optimizer = Adam()
    model = Sequential()
    model.add(inputs)
    model.add(layer1)
    model.add(Dropout(0.5))
    model.add(output)

    call = TensorBoard(log_dir=log_dir, write_grads=True, histogram_freq=1)
    early_stop = EarlyStopping(
        monitor='val_loss',
        mode='min',
        min_delta=0,
        patience=3,
        verbose=1,
        restore_best_weights=True)

    model.compile(optimizer, loss="categorical_crossentropy",
                  metrics=["accuracy"])
    model.fit_generator(
        train_generator, steps_per_epoch=train_size//batch_size, epochs=epochs_num,
        callbacks=[call, early_stop])
#   model.fit_generator(train_generator, steps_per_epoch=5, epochs=5, callbacks=[call])
    model.save(model_dir)
    end = time.time()
    print("Over train job in %f s" % (end-start))

    # test_size=6593
    # input_num=258
    # dims_num=128
    # batch_size=128


def test(model_dir, test_generator, test_size, input_num, dims_num, batch_size):
    model = load_model(model_dir)
    labels_pre = []
    labels_true = []
    batch_num = test_size//batch_size+1
    steps = 0

    for batch, labels in test_generator:
        if len(labels) == batch_size:
            labels_pre.extend(model.predict_on_batch(batch))
        else:
            # batch = np.concatenate((batch, np.zeros(
            #     (batch_size-len(labels), input_num, dims_num.shape[0]))))  # 数组拼接操作

            batch = np.concatenate((batch, np.zeros(
                (batch_size-len(labels), input_num, dims_num))))  # 数组拼接操作

            labels_pre.extend(model.predict_on_batch(batch)[0:len(labels)])
        labels_true.extend(labels)
        steps += 1
        # print("%d/%d batch" % (steps, batch_num))
    labels_pre = np.array(labels_pre).round()

    def to_y(labels):
        y = []
        for i in range(len(labels)):
            if labels[i][0] == 1:
                y.append(0)
            else:
                y.append(1)
        return y
    y_true = to_y(labels_true)
    y_pre = to_y(labels_pre)
    precision = precision_score(y_true, y_pre)
    recall = recall_score(y_true, y_pre)
    print("Precision score is :", precision)
    print("Recall score is :", recall)


def train_and_eval():
    train_generator, test_generator, train_size, test_size, input_num, dims_num = build_dataset(
        batch_size)
    train(train_generator, train_size, input_num, dims_num)
    test(model_dir, test_generator, test_size, input_num, dims_num, batch_size)


if __name__ == "__main__":
    train_and_eval()
