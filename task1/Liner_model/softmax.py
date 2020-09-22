# -*- coding: utf-8 -*-
"""
__title__="softmax"
__author__="ngc7293"
__mtime__="2020/8/31"
"""
import numpy as np
from scipy.special import softmax
from sklearn.metrics import accuracy_score


class softmax_reg():
    def __init__(self, feature_size, target_size):
        self.n_feature = feature_size
        self.n_target = target_size
        self.W = np.random.uniform(size=[self.n_feature, self.n_target], low=-0.1, high=0.1)
        self.b = np.zeros(shape=[1, self.n_target])

    def softmax(self, x):
        exp_x = np.exp(x)
        sum_x = np.sum(np.exp(x), axis=1, keepdims=True)
        return exp_x / sum_x

    def compute_y_hat(self, x):
        y_hat = np.dot(x, self.W) + self.b
        y_hat = self.softmax(y_hat)
        return y_hat

    def compute_loss(self, y, y_hat, n_sample):
        return -(1 / n_sample) * np.sum(y * np.log(y_hat))

    def train(self, data_x, data_y, test_x, test_y, lr=0.05, epoches=40, batch_size=128):
        print("*" * 20+"\n")
        print("training new model which argv is:"
              "learning rate is %f"
              "epoch is %d"
              "batch size is %d\n" % (lr, epoches, batch_size))

        n_sample = data_x.shape[0]

        # y = self.get_onehot(data_y)  # y is targte vector which is [n_sample, n_target]

        test_x = test_x.toarray()

        loss_history = []
        acc_history = []
        for i in range(epoches):

            loss_history_epoch = []

            time = 0
            for start in range(0, n_sample - 1, batch_size):
                time += 1
                if start + batch_size < n_sample - 1:
                    end = start + batch_size
                else:
                    end = n_sample - 1
                    batch_size = n_sample - start
                x = data_x[start:end].toarray()
                y = self.get_onehot(data_y[start:end])

                y_hat = self.compute_y_hat(x)
                loss = self.compute_loss(y, y_hat, batch_size)
                loss_history_epoch.append(loss)

                dw = -(1 / batch_size) * np.dot(x.T, (y - y_hat))  # x is [n_batch_size, self.n_feature] y is [n_batch_size, n_target]
                db = -(1 / batch_size) * np.sum(y - y_hat, axis=0)

                self.W = self.W - lr * dw
                self.b = self.b - lr * db
                # print("\n %d iteration's loss is %f"%(time, loss))

            loss_epoch = np.sum(loss_history_epoch) / len(loss_history_epoch)
            loss_history.append(loss_epoch)

            acc = self.evaluation(test_x, test_y)
            acc_history.append(acc)
            # print("*" * 20)
            # print("\n%d epoches loss is %f" % (i, loss_iter))
            # print("\n       accuracy is %2f" % acc)

        return loss_history, acc_history

    def get_onehot(self, y):
        """
        this function is for y to put a number to target_vector
        :param y: the target number vector
        :return:  the target vector
        """
        res = np.zeros((y.shape[0], self.n_target))
        y = y.astype('int64')
        res[np.arange(y.shape[0]), y.T] = 1
        return res

    def predict(self, x):
        y_pred = self.compute_y_hat(x)
        res = np.argmax(y_pred, axis=1)
        return res

    def evaluation(self, test_x, test_y):
        y_pred = self.predict(test_x)
        score = accuracy_score(list(test_y), list(y_pred))
        return score
