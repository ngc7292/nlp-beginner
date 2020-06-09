# -*- coding: utf-8 -*-
"""
__title__="text_feature.BagOfWord"
__author__="ngc7293"
__mtime__="2020/6/3"
"""

import numpy as np
import pandas as pd
import time
from scipy import sparse
import json

from task1.text_feature.initData import process_sentence
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords


class BagOfWord:
    def __init__(self, stop_words=True):
        self.words_num = 0
        self.words_set = None #word set
        self.words_dict = None

        self.stop_words = stop_words

    def fit(self, f_data: np.array):
        """
        this fit a word bag of all datas
        :param f_data:
        :return: None
        """
        sentence_map = map(lambda x: str(x).split(" "), f_data)
        self.words_set = set()
        for sentence in sentence_map:
            for word in sentence:
                if self.stop_words:
                    if len(word) == 1:
                        continue
                self.words_set.add(word)

        self.words_set = np.sort(list(self.words_set))
        self.words_num = self.words_set.shape[0]
        self.words_dict = {word:label for word,label in zip(self.words_set,np.arange(len(self.words_set)))}

    def transform(self, t_data:np.array):
        rows = []
        cols = []
        data_len = len(t_data)
        for i in range(data_len):
            for word in t_data[i].split(" "):
                if self.stop_words:
                    if len(word) == 1:
                        continue
                rows.append(i)
                cols.append(self.words_dict[word])

        vals = np.ones((len(rows),)).astype(int)

        return sparse.csr_matrix((vals,(rows,cols)),shape=(data_len,self.words_num))

    def fit_transform(self, ft_data:np.array):
        self.fit(ft_data)
        return self.transform(ft_data)

if __name__ == '__main__':
    # data = np.genfromtxt('../data/train.tsv.zip', delimiter="\t")
    # start_time = time.time()
    data = pd.read_csv('../data/train.tsv.zip', sep="\t")
    data = np.array(data)
    sentences = data[:, 2]

    start_time = time.time()
    sentences = np.array([process_sentence(i) for i in sentences])
    print("process time:")
    print(time.time() - start_time)

    with open("../data/sentences.txt","w") as fd:
        fd.write(json.dumps(sentences.tolist()))

    start_time = time.time()
    BOW = BagOfWord()
    BOW.fit_transform(sentences)
    print("fit and transform time:")
    print(time.time()-start_time)
    # print(data[0][0].dtype)
