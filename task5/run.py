# -*- coding: utf-8 -*-
"""
__title__="run"
__author__="ngc7293"
__mtime__="2020/10/7"
"""
from data import Poetry
from model import lstm_char
from tqdm import tqdm

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import fitlog
import os
import fitlog

fitlog.commit(__file__)  # auto commit your codes

# fitlog.add_hyper_in_file (__file__) # record your hyperparameters

fitlog.set_log_dir("logs/")

device = torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')

model_save_path = "./models"
result_save_path = "./result"

data = Poetry()

vocab_size = len(data.word_to_id)

model = lstm_char(vocab_size).to(device)

optimizer = optim.Adam(model.parameters(), lr=0.001)

loss_function = nn.CrossEntropyLoss()

# losses = []
# acces = []

epoches = 200
batch_size = 64

hyper_data = {'batch_size': batch_size,
              'loss function': "cross entropy loss",
              'learning rate': 0.001
              }

fitlog.add_hyper(hyper_data)

best_acc = 0.0
for e in range(epoches):
    loss_history = []
    acc_history = []

    hidden = None
    flag = 0
    for i in tqdm(range(int(len(data.poetrys) / batch_size))):
        model.train()

        x, y = data.next_batch(batch_size)

        x = torch.from_numpy(x).long().to(device)
        y = torch.from_numpy(y).long().to(device)

        y = y.view(-1)

        out, hidden = model(x, device)

        # out = torch.argmax(out,axis=-1)
        # out = out.reshape(batch_size, -1)

        optimizer.zero_grad()
        loss = loss_function(out, y)
        if flag == 0:
            loss.backward(retain_graph=True)
            flag = 1
        else:
            loss.backward()
        optimizer.step()
        loss_history.append(loss.item())

        loss_history.append(loss.item())
        yred = torch.argmax(out, axis=-1)
        acc = np.mean((yred.cpu() == y.cpu()).numpy())
        acc_history.append(acc)

    # losses.append(loss_history)
    # acces.append(acc_history)
    real_loss = float(np.mean(loss_history))
    reak_acc = float(np.mean(acc_history))

    print("epoch:%d, acc:%.6f, loss : %.6f" % (e, real_loss, reak_acc))

    fitlog.add_loss(real_loss, name="Loss", step=e)
    fitlog.add_metric({"dev": {"Acc": reak_acc}}, step=e)
    if np.mean(acc_history) >= best_acc:
        torch.save(model.state_dict(), os.path.join(model_save_path, "epoch_%d_model.pkl" % e))
        best_acc = np.mean(acc_history)


# pd.DataFrame(losses).to_csv(os.path.join(result_save_path, "loss_history_v1.csv"))
# pd.DataFrame(acces).to_csv(os.path.join(result_save_path, "acc_history_v1.csv"))

fitlog.finish()
