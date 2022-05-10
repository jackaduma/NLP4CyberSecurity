
#!python
# -*- coding: utf-8 -*-
# @author: Kun

'''
Author: Kun
Date: 2022-05-10 09:42:38
LastEditTime: 2022-05-10 09:48:00
LastEditors: Kun
Description: 
FilePath: /my_open_projects/NLP4CyberSecurity/utils/eval_utils.py
'''


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


def evaluate_result(y_true, y_pre):
    accuracy = accuracy_score(y_true, y_pre)
    precision = precision_score(y_true, y_pre)
    recall = recall_score(y_true, y_pre)
    f1 = f1_score(y_true, y_pre)
    auc = roc_auc_score(y_true, y_pre)

    print("Accuracy Score is: ", accuracy)
    print("Precision Score is :", precision)
    print("Recall Score is :", recall)
    print("F1 Score: ", f1)
    print("AUC Score: ", auc)


def to_y(labels):
    y = []
    for i in range(len(labels)):
        label = labels[i]
        if label < 0.5:
            y.append(0)

        else:
            y.append(1)

    return y