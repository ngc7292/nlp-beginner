# -*- coding: utf-8 -*-
"""
__title__="word2idx"
__author__="ngc7293"
__mtime__="2020/8/31"
"""
from sklearn.feature_extraction.text import CountVectorizer

import pandas as pd
import numpy as np


class tokenizer():
    def __init__(self):
        train_data = pd.read_csv("../data/process_data_train.csv")

        train_data = np.array(train_data)

        self.train_data_x = train_data[:, 1]
        self.train_data_y = train_data[:, 2]

        self.tokenizer_model = CountVectorizer()

        self.train_data_x = self.tokenizer_model.fit(self.train_data_x)

    def word2idx(self,word):
        if isinstance(word,str):
            word = [word]
        if not isinstance(word,list):
            word = list(word)
        return self.tokenizer_model.transform(word)

if __name__ == '__main__':
    a = tokenizer()
    print(a.word2idx("escapades demonstrating").toarray())



