# -*- coding: utf-8 -*-
"""
__title__="Logistic"
__author__="ngc7293"
__mtime__="2020/7/7"
"""
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score


def sigmoid(z):
    return 1 / (1 - np.exp(-z))


class Logistic_reg:
    def __init__(self, feature_size):
        self.n_feature = feature_size
        self.W = np.random.uniform(size=[self.n_feature, 1],low=-0.1,high=0.1)
        self.b = np.zeros(shape=[1, 1])

    def compute_y_hat(self,x):
        y_hat = np.dot(x,self.W)+self.b
        y_hat = sigmoid(y_hat)
        return y_hat

    def compute_loss(self, y, y_hat, n_sample):
        return -(1/n_sample) * np.sum(y*np.log(y_hat))

    def train(self, data_x, data_y, test_x, test_y, lr=0.05, epoches=40,batch_size=128):
        print("*" * 20 + "\n")
        print("training new model which argv is:"
              "learning rate is %f"
              "epoch is %d"
              "batch size is %d\n" % (lr, epoches, batch_size))

        n_sample = data_x.shape[0]
        test_x = test_x.toarray()

        loss_history = []
        acc_history = []

        for i in range(epoches):
            loss_history_epoch = []
            time = 0
            for start in range(0,n_sample-1,batch_size):
                time+=1
                if start + batch_size < n_sample - 1:
                    end = start + batch_size
                else:
                    end = n_sample - 1
                    batch_size = n_sample - start
                x = data_x[start,end].toarray()
                y = data_y

                y_hat = self.compute_y_hat(x)
                loss = self.compute_loss(y,y_hat,batch_size)

                loss_history_epoch.append(loss)

                dw = -(1 / batch_size) * np.dot(x.T, (y-y_hat))
                db = -(1 / batch_size) * np.sum(y-y_hat,axis=0)

                self.W -= lr*dw
                self.b -= lr*db

            loss_epoch = np.sum(loss_history_epoch)/ len(loss_history_epoch)
            loss_history.append(loss_epoch)

            acc_history.append(self.evaluation(test_x,test_y))

    def predict(self,x):
        y_pred = self.compute_y_hat(x)
        return y_pred

    def evaluation(self, test_x, test_y):
        y_pred = self.compute_y_hat(test_x)
        score = accuracy_score(list(test_y), list(y_pred))
        return score

if __name__ == '__main__':
    pass
