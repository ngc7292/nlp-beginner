# -*- coding: utf-8 -*-
"""
__title__="tf-idf transformer"
__author__="ngc7293"
__mtime__="2020/6/7"
"""
import numpy as np
import json
import time

from scipy import sparse

from initData import process_sentence
from sklearn.preprocessing import normalize # 偷个懒，毕竟我也没有希望做一个新的sklearn（在学了，在学了）


class TfIdf:
    def __init__(self):
        self.word_set = None
        self.word_dict = None
        self.words_num = None
        self.word_num = 0

    def fit(self, f_data: np.array):
        """
        :param f_data:
        :return:
        """
        sentence_map = map(lambda x: str(x).split(" "), f_data)
        self.word_set = set()
        self.words_num = dict()
        for sentence in sentence_map:
            for word in sentence:
                if len(word) == 1:
                    continue
                if word not in self.word_set:
                    self.words_num[word] = 1
                else:
                    self.words_num[word] += 1
                self.word_set.add(word)

        self.word_set = np.sort(list(self.word_set))
        self.words_num = self.word_set.shape[0]
        self.word_dict = {word: label for word, label in zip(self.word_set, np.arange(len(self.word_set)))}

    def transform(self, t_data: np.array):
        rows = []
        cols = []
        data_len = len(t_data)
        for i in range(data_len):
            for word in t_data[i].split(" "):
                if len(word) == 1:
                    continue
                rows.append(i)
                cols.append(self.word_dict[word])

        vals = np.ones((len(rows),)).astype(int)

        csr_matrix = sparse.csr_matrix((vals, (rows, cols)), shape=(data_len, self.words_num))
        words_sum = csr_matrix.sum()
        # tf = np.divide(csr_matrix, words_sum)
        tf = csr_matrix

        # sentence_word_num_list = np.count_nonzero(csr_matrix,axis=0)
        sentence_word_num_list = np.bincount(csr_matrix.indices)
        sentences_num = len(t_data)
        idf = sparse.diags(np.log((sentences_num+1)/(sentence_word_num_list+1))+1,offsets=0,shape=(self.words_num,self.words_num),format='csr')

        return_data = tf*idf
        return normalize(return_data)

    def fit_transform(self,data):
        self.fit(data)
        return self.transform(data)


if __name__ == '__main__':
    with open("../data/sentences.txt", "rb") as fd:
        f_r = fd.read().decode("utf-8")
        data = json.loads(f_r)
    data = np.array(data)

    start_time = time.time()
    #  = TfidfVectorizer()
    # retuen_data = data_model.fit_transform(data)
    # print(retuen_data.shape)
    # print(time.time() - start_time)
    # corpus = [
    #     'This is the first document.',
    #     'This document is the second document.',
    #     'And this is the third one.',
    #     'Is this the first document?',
    # ]

    # corpus = np.array([process_sentence(s) for s in corpus])
    data_model = TfIdf()
    return_data = data_model.fit_transform(data)

    print(return_data.shape)
    print(time.time()-start_time)
    # print(data_model.word_dict)