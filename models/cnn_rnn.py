#!python
# -*- coding: utf-8 -*-
# @author: Kun

'''
Author: Kun
Date: 2022-03-18 16:00:33
LastEditTime: 2022-05-09 22:21:57
LastEditors: Kun
Description: 
FilePath: /my_open_projects/NLP4CyberSecurity/models/cnn_rnn.py
'''

from keras.models import Sequential, Model
from keras import regularizers
from keras.layers.core import Dense, Dropout, Activation, Lambda, Flatten
from keras.layers import Input, ELU, LSTM, Embedding, Convolution2D, MaxPooling2D, \
    BatchNormalization, Convolution1D, MaxPooling1D, concatenate
from keras.preprocessing import sequence
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
from keras import backend as K


class ConvLSTM(object):
    def __init__(self) -> None:
        super(ConvLSTM, self).__init__()

        self.max_len = 75
        self.emb_dim = 32
        self.max_vocab_len = 100
        self.lstm_output_size = 32
        self.W_reg = regularizers.l2(1e-4)

    def build_model(self):
        main_input = Input(shape=(self.max_len,),
                           dtype='int32', name='main_input')
        emb = Embedding(input_dim=self.max_vocab_len, output_dim=self.emb_dim, input_length=self.max_len,
                        embeddings_regularizer=self.W_reg)(main_input)
        emb = Dropout(0.25)(emb)

        conv = Convolution1D(kernel_size=5, filters=256,
                             padding='same')(emb)
        conv = ELU()(conv)

        conv = MaxPooling1D(pool_size=4)(conv)
        conv = Dropout(0.5)(conv)

        lstm = LSTM(self.lstm_output_size)(conv)
        lstm = Dropout(0.5)(lstm)

        output = Dense(1, activation='sigmoid', name='output')(lstm)

        model = Model(inputs=[main_input], outputs=[output])

        adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999,
                    epsilon=1e-08, decay=0.0)
        model.compile(optimizer=adam, loss='binary_crossentropy',
                      metrics=['accuracy'])
        return model
