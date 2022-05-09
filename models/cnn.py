#!python
# -*- coding: utf-8 -*-
# @author: Kun

'''
Author: Kun
Date: 2022-03-18 16:00:18
LastEditTime: 2022-05-09 22:21:53
LastEditors: Kun
Description: 
FilePath: /my_open_projects/NLP4CyberSecurity/models/cnn.py
'''


import tensorflow as tf
from keras.models import Sequential, Model
from keras import regularizers
from keras.layers.core import Dense, Dropout, Activation, Lambda, Flatten
from keras.layers import Input, ELU, LSTM, Embedding, Convolution2D, MaxPooling2D, \
    BatchNormalization, Convolution1D, MaxPooling1D, concatenate
from keras.preprocessing import sequence
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
from keras import backend as K


class ConvFully(object):
    def __init__(self) -> None:
        super(ConvFully, self).__init__()
        self.max_len = 75
        self.emb_dim = 32
        self.max_vocab_len = 100
        self.W_reg = regularizers.l2(1e-4)

    def sum_1d(self, X):
        return K.sum(X, axis=1)

    def get_conv_layer(self, emb, kernel_size=5, filters=256):
        conv = Convolution1D(kernel_size=kernel_size, filters=filters,
                             padding='same')(emb)
        conv = ELU()(conv)

        conv = Lambda(self.sum_1d, output_shape=(filters,))(conv)

        # conv = BatchNormalization()(conv)

        conv = Dropout(0.5)(conv)
        return conv

    def build_model(self):
        main_input = Input(shape=(self.max_len,),
                           dtype='int32', name='main_input')
        emb = Embedding(input_dim=self.max_vocab_len, output_dim=self.emb_dim, input_length=self.max_len,
                        embeddings_regularizer=self.W_reg)(main_input)
        emb = Dropout(0.25)(emb)

        conv1 = self.get_conv_layer(emb, kernel_size=2, filters=256)
        conv2 = self.get_conv_layer(emb, kernel_size=3, filters=256)
        conv3 = self.get_conv_layer(emb, kernel_size=4, filters=256)
        conv4 = self.get_conv_layer(emb, kernel_size=5, filters=256)

        merged = concatenate([conv1, conv2, conv3, conv4], axis=1)

        hidden1 = Dense(1024)(merged)
        hidden1 = ELU()(hidden1)
        hidden1 = BatchNormalization()(hidden1)
        hidden1 = Dropout(0.5)(hidden1)

        hidden2 = Dense(1024)(hidden1)
        hidden2 = ELU()(hidden2)
        hidden2 = BatchNormalization()(hidden2)
        hidden2 = Dropout(0.5)(hidden2)

        output = Dense(1, activation='sigmoid', name='output')(hidden2)

        model = Model(inputs=[main_input], outputs=[output])

        adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999,
                    epsilon=1e-08, decay=0.0)
        model.compile(optimizer=adam, loss='binary_crossentropy',
                      metrics=['accuracy'])
        return model
