from sklearn.feature_extraction.text import CountVectorizer

import numpy as np
import pandas as pd
import os
import time

from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer


def process_string(string,
                   remove_no_letter=True,
                   is_lower=True,
                   is_lemmatizer=True,
                   remove_stop_word=True):
    """
    this function is for process string using such tag
    :param string: orgin string
    :param remove_no_letter: target of if remove no letters
    :param is_lower: target of reset every word to lower case
    :param is_lemmatizer: target of if make the word lemmatization
    :param remove_stop_word: target of if remove stop word
    :return: the string which has processed.
    """

    res_string = BeautifulSoup(string, 'lxml').get_text()

    if remove_no_letter:
        res_string = re.sub('[^a-zA-Z]', ' ', res_string)

    if is_lower:
        res_string = res_string.lower().split()

    if is_lemmatizer:
        lemmatizer = WordNetLemmatizer()
        res_string = set([lemmatizer.lemmatize(str(x)) for x in res_string])

    if remove_stop_word:
        stop_word_set = set(stopwords.words('english'))
        res_string = " ".join([i for i in res_string if not i in stop_word_set])

    return res_string


class data():
    def __init__(self, dic_path='../data'):
        self.dic_path = dic_path
        train_data, test_data = self.load_data()

        if not self.check_path("process_data_train.csv"):
            train_data = self.process_data(train_data)
            self.save_data("process_data_train.csv", train_data)
        else:
            train_data = self.read_data("process_data_train.csv")

        if not self.check_path("process_data_test.csv"):
            test_data = self.process_data(test_data)
            self.save_data("process_data_test.csv", test_data)
        else:
            test_data = self.read_data("process_data_test.csv")

        self.train_data = train_data
        self.test_data = test_data

        self.model = CountVectorizer()

    def load_data(self):
        train_path = os.path.join(self.dic_path, 'train.tsv')
        test_path = os.path.join(self.dic_path, 'test.tsv')

        train_data = np.array(pd.read_csv(train_path, sep='\t'))
        test_data = np.array(pd.read_csv(test_path, sep='\t'))

        train_data = train_data[:, 2:]
        test_data = test_data[:, 2:]

        return train_data, test_data

    def process_data(self, data, is_train=True):
        """
        this function is for get process_data which use different process
        :param data:
        :return:
        """
        if is_train:
            data_x = data[:, 0]
            data_y = data[:, 1]

            data_x_res = np.array([process_string(s) for s in data_x])
            data_x_res.reshape(-1, 1)
            data_y.reshape(-1, 1)

            data_res = np.concatenate((data_x_res, data_y), axis=-1)
        else:
            data = data[:, 0]
            data = np.array([process_string(s) for s in data])
            data.reshape(-1, 1)
            data_res = data

        return data_res

    def check_path(self, file_name=""):
        if os.path.exists(os.path.join(self.dic_path, file_name)):
            return True
        else:
            return False

    def save_data(self, file_name, data):
        pd.DataFrame(data).to_csv(os.path.join(self.dic_path, file_name))

    def read_data(self, file_name):
        read_data = pd.read_csv(os.path.join(self.dic_path, file_name))
        return np.array(read_data)

    def get_data(self):
        return self.train_data, self.test_data

    def init_model(self, type='unigram'):
        if type == "unigram":
            self.model = CountVectorizer(min_df=0.0005, max_df=0.9995, ngram_range=(1, 1))
        elif type == "bigram":
            self.model = CountVectorizer(min_df=0.0005, max_df=0.9995, ngram_range=(2, 2))
        elif type == "trigram":
            self.model = CountVectorizer(min_df=0.0005, max_df=0.9995, ngram_range=(3, 3))

    def fit_model(self):
        self.model.fit(self.train_data[:, 1])

    def get_model(self):
        return self.model


if __name__ == '__main__':
    start = time.time()
    a = data()
    train_data, test_data = a.get_data()
    print(time.time() - start)
    # start = time.time()
    # data().process_data(train_data)
    # print(time.time()-start)

    train_x = train_data[:, 1]
    train_y = train_data[:, 2]
    #
    print(train_data)
    print(train_x)
    print(train_y)
