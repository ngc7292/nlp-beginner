# -*- coding: utf-8 -*-
"""
__title__="chekc"
__author__="ngc7293"
__mtime__="2020/9/20"
"""
import os
import numpy as np
import pandas as pd
import time
from tqdm import tqdm

from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors, Word2Vec

glove_file = "../data/pre/glove.42B.300d.zip"
word2vec_file = "../data/pre/glove.42B.300d.word2vec.txt"
#
start_time = time.time()
if not os.path.exists(word2vec_file):
    (count, dim) = glove2word2vec(glove_file,word2vec_file)
print(time.time()-start_time)
#
# glove_model = KeyedVectors.load_word2vec_format(word2vec_file, binary=False)
# print(time.time()-start_time)
#
# start_time = time.time()
# cat_vec = glove_model['csfkjaf']
#
# print(cat_vec)
#
# print(time.time()-start_time)

# df = pd.read_csv("../data/train.tsv",sep="\t")
#
# corpus = []
# for i in tqdm(range(len(df))):
#     corpus.append(df.Phrase[i].lower().split())
#
# word2vec_model = Word2Vec(corpus,size=100)
# word2vec_wv = word2vec_model.wv
#
# word2vec_wv.save_word2vec_format("../data/pre/word2vec.100d.word2vec.txt",binary=False)