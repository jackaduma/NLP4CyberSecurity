#!python
# -*- coding: utf-8 -*-
# @author: Kun

'''
Author: Kun
Date: 2022-06-09 10:57:32
LastEditTime: 2022-06-09 10:57:36
LastEditors: Kun
Description: 
FilePath: /my_open_projects/NLP4CyberSecurity/models/simple_bilstm.py
'''

import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from keras.models import Sequential
from keras.layers import LSTM, GRU, Embedding, Dense, Flatten, Bidirectional
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization


def build_tokenizer(samples):
    max_chars = 20000
    maxlen = 128

    tokenizer = Tokenizer(num_words=max_chars, char_level=True)
    tokenizer.fit_on_texts(samples)
    sequences = tokenizer.texts_to_sequences(samples)
    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))

    num_chars = len(tokenizer.word_index)+1
    embedding_vector_length = 128

    return max_chars, maxlen, num_chars, embedding_vector_length, sequences


def build_model(num_chars, embedding_vector_length, maxlen):
    # Create model for training.
    model = Sequential()
    model.add(Embedding(num_chars, embedding_vector_length, input_length=maxlen))
    model.add(Bidirectional(
        LSTM(256, dropout=0.3, recurrent_dropout=0.3, return_sequences=True)))
    model.add(Bidirectional(
        LSTM(256, dropout=0.3, recurrent_dropout=0.3, return_sequences=True)))
    model.add(Bidirectional(LSTM(128, dropout=0.3, recurrent_dropout=0.3)))
    model.add(Dense(1, activation='sigmoid'))

    model.summary()

    model.compile(optimizer='adam', loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model
