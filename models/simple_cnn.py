#!python
# -*- coding: utf-8 -*-
# @author: Kun


import os
import time
from keras.models import Sequential
from keras.layers import Dense, InputLayer, Dropout, LSTM, Conv1D, Flatten, GlobalAveragePooling1D, MaxPool1D
from keras.callbacks import EarlyStopping, TensorBoard
from keras.optimizers import Adam

from data_loader.xss_datasets import build_dataset
from utils import init_session
from models.simple_nn import test

CACHE_DIR = "./cache/xss_injection"

init_session()

batch_size = 8  # 500
epochs_num = 50  # 1
process_datas_dir = os.path.join(CACHE_DIR, "process_datas.pickle")

log_dir = "log/simple_cnn.log"
model_dir = os.path.join(CACHE_DIR, "simple_cnn_model")


def train(train_generator, train_size, input_num, dims_num):
    print("Start Train Job! ")
    start = time.time()
    inputs = InputLayer(input_shape=(input_num, dims_num),
                        batch_size=batch_size)
    layer1 = Conv1D(64, 3, activation="relu")
    layer2 = Conv1D(64, 3, activation="relu")
    layer3 = Conv1D(128, 3, activation="relu")
    layer4 = Conv1D(128, 3, activation="relu")
    layer5 = Dense(128, activation="relu")
    output = Dense(2, activation="softmax", name="Output")
    optimizer = Adam()
    model = Sequential()
    model.add(inputs)
    model.add(layer1)
    model.add(layer2)
    model.add(MaxPool1D(pool_size=2))
    model.add(Dropout(0.5))
    model.add(layer3)
    model.add(layer4)
    model.add(MaxPool1D(pool_size=2))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(layer5)
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
    model.fit_generator(train_generator, steps_per_epoch=train_size //
                        batch_size, epochs=epochs_num,
                        callbacks=[call, early_stop])
#    model.fit_generator(train_generator, steps_per_epoch=5, epochs=5, callbacks=[call])
    model.save(model_dir)
    end = time.time()
    print("Over train job in %f s" % (end-start))


def train_and_eval():
    train_generator, test_generator, train_size, test_size, input_num, dims_num = build_dataset(
        batch_size)
    train(train_generator, train_size, input_num, dims_num)
    test(model_dir, test_generator, test_size, input_num, dims_num, batch_size)


if __name__ == "__main__":
    train_and_eval()
