# -*- coding: utf-8 -*-
"""
__title__="run"
__author__="ngc7293"
__mtime__="2020/9/25"
"""

from data import get_iter
from model import ESIM
import numpy as np
import pandas as pd

import torch
from tqdm import tqdm
from torch import optim
import torch.nn as nn

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
data_path = "../data/snli/snli_1.0/"
train_iter, dev_iter, test_iter, TEXT, LABEL = get_iter(data_path, device, 32)

model = ESIM(len(TEXT.vocab), len(LABEL.vocab.stoi), 300, 600, 0.5, 1, TEXT.vocab.vectors).to(device)


optimizer = optim.Adam(model.parameters(),lr=0.001)
loss_function = nn.CrossEntropyLoss()


loss_history = []
test_acc_history = []


for e in tqdm(range(20)):
    model.train()
    losses = []
    count = 0
    print("traing models %d epoches ... "%e)
    for data in tqdm(train_iter):
        x1, x1_lens = data.x0
        x2, x2_lens = data.x1
        y = data.y.squeeze(1)

        y_pred = model(x1, x1_lens, x2, x2_lens)

        optimizer.zero_grad()
        loss = loss_function(y_pred,y)
        loss.backward()
        losses.append(loss.item())
        optimizer.step()

    train_loss = np.mean(losses)
    loss_history.append(losses)

    model.eval()

    # train_acces = []
    # for data in train_iter:
    #     x1, x1_lens = data.x0
    #     x2, x2_lens = data.x1
    #     y = data.y.squeeze(1).numpy()
    #
    #     y_pred = np.argmax(model(x1,x1_lens,x2,x2_lens).cpu().detach().numpy(), 1)
    #
    #     acc = np.mean(y_pred == y,dtype=np.float)
    #
    #     train_acces.append(acc)
    #
    # train_acc = np.mean(train_acces)
    # train_acc_history.append(train_acc)

    test_acces = []
    for data in dev_iter:
        x1, x1_lens = data.x0
        x2, x2_lens = data.x1
        y = data.y.cpu().squeeze(1).numpy()

        y_pred = np.argmax(model(x1, x1_lens, x2, x2_lens).cpu().detach().numpy(), 1)

        acc = np.mean(y_pred == y, dtype=np.float)

        test_acces.append(acc)

    test_acc = np.mean(test_acces)
    test_acc_history.append(test_acc)

    print("epoch %d train loss:%.6f, test acc:%.7f" % (
        e, float(train_loss), float(test_acc)))

    model_path = "models/model_glove_esim_%d_epoches.bin"%e

    torch.save(model, model_path)

pd.DataFrame(loss_history).to_csv("./result/loss_history_100d.csv")
    # pd.DataFrame(train_acc_history).to_csv("./result/acc_history_100d.csv")
pd.DataFrame(test_acc_history).to_csv("./result/test_acc_history_100d.csv")

















