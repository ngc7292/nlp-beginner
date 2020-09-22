# -*- coding: utf-8 -*-
"""
__title__="train"
__author__="ngc7293"
__mtime__="2020/9/20"
"""
from data_load import get_dataloader
from model import TextCnn, TextRnn
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim


if __name__ == '__main__':
    lr = 0.001
    epoches = 50
    class_num = 5
    batch_size = 32
    sentence_len = 40

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    train_loader, test_loader, embed_pretrain_glove, vocab_size = get_dataloader(sentence_len=sentence_len, pre_model="glove")

    _, _, embed_pretrain_word2vec, _ = get_dataloader(sentence_len=sentence_len, pre_model="word2vec")
    rnn_model_with_glove = TextRnn(vocab_size=vocab_size, embedding_dim=50, class_num=5, hidden_size=128,
                                   embedding_pretrained=embed_pretrain_glove, device=device)

    cnn_model_with_glove = TextCnn(vocab_size=vocab_size, embedding_dim=50, hidden_size=128, class_num=class_num,
                                   embedding_pretrained=embed_pretrain_glove)

    rnn_model_without = TextRnn(vocab_size=vocab_size, embedding_dim=50, class_num=5, hidden_size=128,
                                embedding_pretrained=None, device=device)

    cnn_model_without = TextCnn(vocab_size=vocab_size, embedding_dim=50, hidden_size=128, class_num=class_num,
                                embedding_pretrained=None)

    rnn_model_with_word2vec = TextRnn(vocab_size=vocab_size, embedding_dim=50, class_num=5, hidden_size=128,
                                embedding_pretrained=embed_pretrain_word2vec, device=device)

    cnn_model_with_word2vec =  TextCnn(vocab_size=vocab_size, embedding_dim=50, hidden_size=128, class_num=class_num,
                                embedding_pretrained=embed_pretrain_word2vec)

    model_list = [rnn_model_with_glove,
                  rnn_model_without,
                  rnn_model_with_word2vec,
                  cnn_model_with_glove,
                  cnn_model_without,
                  cnn_model_with_word2vec]

    model_type_list = [
        "rnn with glove",
        "rnn with random wmbedding",
        "rnn with word2vec",
        "cnn with glove",
        "cnn with random embedding",
        "cnn with word2vec"
    ]
    # model_list = [rnn_model_with_word2vec]

    loss_history_list = []
    train_acc_list = []
    test_acc_list = []

    for model,model_type in zip(model_list,model_type_list):

        # if model_type == "cnn": model = TextCnn(vocab_size=vocab_size, embedding_dim=50, hidden_size=128,
        # class_num=class_num,  embedding_pretrained=embed_pretrain)

        print("traing %s model ..."%model_type)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        loss_function = nn.CrossEntropyLoss()

        loss_history = []
        train_acc_history = []
        test_acc_history = []

        model.to(device)
        for epoch in range(epoches):
            model.train()
            losses = []
            count = 0
            for idx, (x, y) in enumerate(train_loader):
                x, y = x.to(device), y.to(device)
                # if len(x) < batch_size:
                #     break
                pred = model(x)
                optimizer.zero_grad()
                loss = loss_function(pred, y)
                loss.backward()
                losses.append(loss.item())
                optimizer.step()

            train_loss = np.mean(losses)
            loss_history.append(train_loss)

            # model.eval() will notify all your layers that you are in eval mode, that way, batchnorm or dropout
            # layers will work in eval mode instead of training mode. torch.no_grad() impacts the autograd engine and
            # deactivate it. It will reduce memory usage and speed up computations but you won’t be able to backprop
            # (which you don’t want in an eval script).

            model.eval()

            train_acces = []

            for idx, (x, y) in enumerate(train_loader):
                x = x.to(device)
                y = y.numpy()
                y_pred = np.argmax(model(x).cpu().detach().numpy(), 1)

                acc = np.mean(y_pred == y, dtype=np.float)
                train_acces.append(acc)
            train_acc = np.mean(train_acces)
            train_acc_history.append(train_acces)

            test_acces = []
            for idx, (x, y) in enumerate(test_loader):
                x = x.to(device)
                y = y.numpy()
                y_pred = np.argmax(model(x).cpu().detach().numpy(), 1)

                acc = np.mean(y_pred == y, dtype=np.float)
                test_acces.append(acc)

            test_acc = np.mean(test_acces)
            test_acc_history.append(test_acces)

            print("epoch %d train loss:%.6f,train acc:%.6f test acc:%.7f" % (
            epoch, float(train_loss), float(train_acc), float(test_acc)))

        loss_history_list.append(loss_history)
        train_acc_list.append(train_acc_history)
        test_acc_list.append(test_acc_history)

    pd.DataFrame(loss_history_list).to_csv("./result/loss_history.csv")
    pd.DataFrame(train_acc_list).to_csv("./result/acc_history.csv")
    pd.DataFrame(test_acc_list).to_csv("./result/test_acc_history.csv")


