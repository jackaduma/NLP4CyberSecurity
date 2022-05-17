#!python
# -*- coding: utf-8 -*-
# @author: Kun

'''
Author: Kun
Date: 2022-05-10 09:42:15
LastEditTime: 2022-05-16 19:57:59
LastEditors: Kun
Description: 
FilePath: /my_open_projects/NLP4CyberSecurity/utils/__init__.py
'''
import tensorflow as tf

# import keras.backend.tensorflow_backend as ktf


def init_session():
    # gpu_options=tf.GPUOptions(allow_growth=True)
    # ktf.set_session(tf.Session())  # 创建一个会话，当上下文管理器退出时会话关闭和资源释放自动完成。
    tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session())
