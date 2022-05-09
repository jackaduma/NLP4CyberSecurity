#!python
# -*- coding: utf-8 -*-
# @author: Kun

'''
Author: Kun
Date: 2022-05-09 15:28:02
LastEditTime: 2022-05-09 15:28:05
LastEditors: Kun
Description: 
FilePath: /my_open_projects/weak-password-detection-with-ML/models/svm_clf.py
'''


from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report


def train_and_eval(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.10, random_state=42)  # splitting

    # our logistic regression classifier
    # lgs = LogisticRegression(penalty='l2', multi_class='ovr', n_jobs=1)
    # lgs.fit(X_train, y_train)  # training
    # print(lgs.score(X_test, y_test))  # testing

    model = LinearSVC()
    # model = SVC(probability=True, kernel="linear")
    # model = RandomForestClassifier(n_estimators=250,max_features=0.2)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    print(classification_report(y_test, pred, digits=5))

    return model


def eval(vectorizer, model):
    # more testing
    X_predict = ['faizanahmad', 'faizanahmad123', 'faizanahmad##', 'ajd1348#28t**', 'ffffffffff', 'kuiqwasdi',
                 'uiquiuiiuiuiuiuiuiuiuiuiui', 'mynameisfaizan', 'mynameis123faizan#', 'faizan', '123456', 'abcdef']
    X_predict = vectorizer.transform(X_predict)
    y_Predict = model.predict(X_predict)
    print(y_Predict)
