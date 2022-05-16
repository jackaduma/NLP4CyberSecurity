#!python
# -*- coding: utf-8 -*-
# @author: Kun

'''
Author: Kun
Date: 2022-05-16 14:41:17
LastEditTime: 2022-05-16 14:41:41
LastEditors: Kun
Description: 
FilePath: /my_open_projects/NLP4CyberSecurity/utils/seg_util.py
'''

import nltk
import re
from urllib.parse import unquote


def GeneSeg(payload):
    payload = payload.lower()  # 变小写
    payload = unquote(unquote(payload))  # 解码
    payload, num = re.subn(r'\d+', "0", payload)  # 数字泛化为"0"
    # 替换url为”http://u
    payload, num = re.subn(
        r'(http|https)://[a-zA-Z0-9\.@&/#!#\?]+', "http://u", payload)
    # 分词
    r = '''
        (?x)[\w\.]+?\(
        |\)
        |"\w+?"
        |'\w+?'
        |http://\w
        |</\w+>
        |<\w+>
        |<\w+
        |\w+=
        |>
        |[\w\.]+
    '''
    return nltk.regexp_tokenize(payload, r)