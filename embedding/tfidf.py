#!python
# -*- coding: utf-8 -*-
# @author: Kun


'''
Author: Kun
Date: 2022-05-09 15:27:04
LastEditTime: 2022-05-09 22:22:10
LastEditors: Kun
Description: 
FilePath: /my_open_projects/NLP4CyberSecurity/embedding/tfidf.py
'''


import pandas as pd
import numpy as np
import random


from sklearn.feature_extraction.text import TfidfVectorizer



def get_tokens(input_string):
    # custom tokenizer. ours tokens are characters rather than full words
    tokens = []
    for i in input_string:
        tokens.append(i)
    return tokens


def make_vector(filepath):
    data = pd.read_csv(filepath, ',', error_bad_lines=False, encoding="utf-8")

    df = pd.DataFrame(data)
    print(df)

    # x = df['password'] == np.nan
    # print(x.to_csv('FindNaN.csv', sep=',', na_rep = 'string', index=True))
    # print(df.isnull().values.any())

    # TODO 因为 password 字段里面 可能有 不是string的内容，需要做一次转换，否则在后续的vector中报错
    df['password'] = df['password'].apply(lambda x: np.str(x)) 

    passwords = np.array(df)

    # shuffling randomly for robustness
    random.shuffle(passwords)

    y = [d[1] for d in passwords]  # labels
    allpasswords = [d[0] for d in passwords]  # actual passwords

    vectorizer = TfidfVectorizer(ngram_range=(2, 4), stop_words=[], min_df=3, max_df=0.8, lowercase=False,
                                 tokenizer=get_tokens, decode_error='replace', encoding="utf-8")  # vectorizing
    X = vectorizer.fit_transform(allpasswords)

    return X, y, vectorizer

    

    

    
