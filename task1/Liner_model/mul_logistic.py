# -*- coding: utf-8 -*-
"""
__title__="mul_logistic"
__author__="ngc7293"
__mtime__="2020/9/9"
"""

from .Logistic import Logistic_reg

import numpy as np
import pandas as pd

class mul_logistic_reg():
    def __init__(self, feature_size, target_size):
        self.n_feature = feature_size
        self.n_target = target_size


    def data_process(self):
        pass

    def train(self, data_x, data_y, test_x, test_y, lr=0.05,epoches=40,batch_size=128):
        pass
