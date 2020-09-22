# -*- coding: utf-8 -*-
"""
__title__="main"
__author__="ngc7293"
__mtime__="2020/7/8"
"""
from .data_process import data
from .Liner_model.softmax import softmax_reg
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from collections import Counter

import pandas as pd

data = data()
train_data, test_data = data.get_data()
# unigram
unigram_model = CountVectorizer(min_df=0.005, max_df=0.995, ngram_range=(1, 1))

# bigram
bigram_model = CountVectorizer(min_df=0.005, max_df=0.995, ngram_range=(2, 2))

# trigram
trigram_model = CountVectorizer(min_df=0.0001, max_df=0.9999, ngram_range=(3, 3))

train_data_x_ori = train_data[:, 1]
train_data_y_ori = train_data[:, 2]

unigram_model.fit(train_data_x_ori)
bigram_model.fit(train_data_x_ori)
trigram_model.fit(train_data_x_ori)

data_models = [unigram_model, bigram_model, trigram_model]

arg_list = [{
    'lr': 0.005,
    'batch': 10000
}, {
    'lr': 0.005,
    'batch': 128
}, {
    'lr': 0.005,
    'batch': 64,
},{
    'lr': 0.005,
    'batch': 32,
}, {
    'lr': 0.005,
    'batch': 16,
},{
    'lr': 0.005,
    'batch': 1
}, {
    'lr': 0.05,
    'batch': 10000
},{
    'lr': 0.05,
    'batch': 128
},{
    'lr': 0.05,
    'batch': 64
},{
    'lr': 0.05,
    'batch': 32
},{
    'lr': 0.05,
    'batch': 16,
},{
    'lr': 0.05,
    'batch': 1
}, {
    'lr': 0.5,
    'batch': 10000
}, {
    'lr': 0.5,
    'batch': 128
},{
    'lr': 0.5,
    'batch': 64
},{
    'lr': 0.5,
    'batch': 32
},{
    'lr': 0.5,
    'batch': 16
},{
    'lr': 0.5,
    'batch': 1
},{
    'lr': 1,
    'batch': 10000
},{
    'lr': 1,
    'batch': 128
},{
    'lr': 1,
    'batch': 64
},{
    'lr': 1,
    'batch': 32
},{
    'lr': 1,
    'batch': 16
},{
    'lr': 1,
    'batch': 1
}]

loss_df = []
acc_df = []


for arg in arg_list:
    data_model_idx = 1
    for data_model in data_models:
        train_data_x = data_model.transform(train_data_x_ori)
        train_data_y = train_data_y_ori

        x_train, x_test, y_train, y_test = train_test_split(train_data_x, train_data_y, test_size=0.33, random_state=33)


        print(Counter(y_train))

        n_sample, n_feature = x_train.shape
        n_target = 5

        softmax_model = softmax_reg(n_feature, n_target)
        loss, acc = softmax_model.train(train_data_x, train_data_y,x_test,y_test,lr=arg['lr'],batch_size=arg['batch'])
        loss_df.append([arg['lr'],arg['batch'],data_model_idx] + loss)
        acc_df.append([arg['lr'],arg['batch'],data_model_idx] + acc)
        data_model_idx += 1

loss_df = pd.DataFrame(data=loss_df)
loss_df.to_csv("./result/loss.csv")
acc_df = pd.DataFrame(data=acc_df)
acc_df.to_csv("./result/acc.csv")





