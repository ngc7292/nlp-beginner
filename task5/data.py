# -*- coding: utf-8 -*-
"""
__title__="data"
__author__="ngc7293"
__mtime__="2020/10/7"
"""

import numpy as np
from io import open
import sys
import collections

class Poetry:
    def __init__(self):
        self.filename = "../data/poems.txt"
        self.poetrys = self.get_poetrys()
        self.poetry_vectors,self.word_to_id,self.id_to_word = self.gen_poetry_vectors()
        self.poetry_vectors_size = len(self.poetry_vectors)
        self._index_in_epoch = 0

    def get_poetrys(self):
        poetrys = list()
        f = open(self.filename,"r", encoding='utf-8')
        for line in f.readlines():
            _,content = line.strip('\n').strip().split(':')
            content = content.replace(' ','')
            if(not content or '_' in content or '(' in content or '（' in content or "□" in content
                   or '《' in content or '[' in content or ':' in content or '：'in content):
                continue

            if len(content) < 5 or len(content) > 79:
                continue
            content_list = content.replace('，', '|').replace('。', '|').split('|')
            flag = True

            for sentence in content_list:
                slen = len(sentence)
                if 0 == slen:
                    continue
                if 5 != slen and 7 != slen:
                    flag = False
                    break
            if flag:
                #每首古诗以'['开头、']'结尾
                poetrys.append('[' + content + ']')
        return poetrys

    def gen_poetry_vectors(self):
        words = sorted(set(''.join(self.poetrys) + ' '))
        #数字ID到每个字的映射
        id_to_word = {i: word for i, word in enumerate(words)}
        #每个字到数字ID的映射
        word_to_id = {v: k for k, v in id_to_word.items()}
        to_id = lambda word: word_to_id.get(word)
        #唐诗向量化
        poetry_vectors = [list(map(to_id, poetry)) for poetry in self.poetrys]
        return poetry_vectors,word_to_id,id_to_word

    def next_batch(self,batch_size):
        assert batch_size < self.poetry_vectors_size
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        #取完一轮数据，打乱唐诗集合，重新取数据
        if self._index_in_epoch > self.poetry_vectors_size:
            np.random.shuffle(self.poetry_vectors)
            start = 0
            self._index_in_epoch = batch_size
        end = self._index_in_epoch
        batches = self.poetry_vectors[start:end]
        x_batch = np.full((batch_size, max(map(len, batches))), self.word_to_id[' '], np.int32)
        for row in range(batch_size):
            x_batch[row,:len(batches[row])] = batches[row]
        y_batch = np.copy(x_batch)
        y_batch[:,:-1] = x_batch[:,1:]
        y_batch[:,-1] = x_batch[:, 0]

        return x_batch,y_batch

if __name__ == '__main__':
    a = Poetry()
    x,y = a.next_batch(32)
    print(x[0])
    print(y[0])
    for i,j in zip(x,y):
        res = []
        res_y = []
        for z,x in zip(i,j):
            res.append(a.id_to_word[z])
            res_y.append(a.id_to_word[x])

        print("".join(res))
        print("".join(res_y))

    print(a.word_to_id["含"])