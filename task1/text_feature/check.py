# -*- coding: utf-8 -*-
"""
__title__=""
__author__="ngc7293"
__mtime__="2020/6/9"
"""
from sklearn.feature_extraction.text import TfidfVectorizer
import json
import time
import numpy as np
from initData import process_sentence

# with open("../data/sentences.txt", "rb") as fd:
#     f_r = fd.read().decode("utf-8")
#     data = json.loads(f_r)
# data = np.array(data)
#
# start_time = time.time()
# data_model = TfidfVectorizer()
# retuen_data = data_model.fit_transform(data)

corpus = [
    'This is the first document.',
    'This document is the second document.',
    'And this is the third one.',
    'Is this the first document?',
]

corpus = np.array([process_sentence(s) for s in corpus])

data_model = TfidfVectorizer()
return_data = data_model.fit_transform(corpus)

print(return_data.toarray())

