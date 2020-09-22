# -*- coding: utf-8 -*-
"""
__title__="model"
__author__="ngc7293"
__mtime__="2020/9/20"
"""

import torch.nn as nn
import torch
import torch.nn.functional as F



class TextCnn(nn.Module):
    def __init__(self,
                 vocab_size,
                 embedding_dim,
                 class_num,
                 embedding_pretrained=None,
                 kernel_size = [3, 4, 5],
                 hidden_size = 100,
                 dropout=0.5,
                 sentence_len = 50):
        super(TextCnn, self).__init__()
        if embedding_pretrained is None:
            self.embed_layer = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=sentence_len)
        else:
            self.embed_layer = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim,
                                            _weight=embedding_pretrained, padding_idx=sentence_len)

        self.conv_layer = nn.ModuleList([nn.Conv2d(1, hidden_size, (k, embedding_dim)) for k in kernel_size])

        self.dropout_layer = nn.Dropout(dropout)
        self.linear_layer = nn.Linear(3*hidden_size, class_num)

    def forward(self,x):
        """
        :param x: batch_size , sentence_len
        :return:
        """
        embed_out = self.embed_layer(x).unsqueeze(1)
        conv_out = [F.relu(conv(embed_out)).squeeze(3) for conv in self.conv_layer]
        pool_out = torch.cat([F.max_pool1d(out, out.size(2)).squeeze(2) for out in conv_out],1)

        out = self.linear_layer(pool_out)
        return out

class TextRnn(nn.Module):
    def __init__(self,
                 vocab_size,
                 embedding_dim,
                 class_num,
                 hidden_size,
                 embedding_pretrained=None,
                 dropout = 0.5,
                 max_len = 50,
                 device = torch.device('cpu')):
        super(TextRnn, self).__init__()

        self.vocab_size =  vocab_size
        self.embedding_dim = embedding_dim
        self.class_num = class_num
        self.hidden_size = hidden_size
        self.device = device

        if embedding_pretrained is None:
            self.embedding_layer = nn.Embedding(vocab_size,embedding_dim,padding_idx=max_len)
        else:
            self.embedding_layer = nn.Embedding(vocab_size, embedding_dim,padding_idx=max_len, _weight=embedding_pretrained)

        self.rnn_layer = nn.RNN(input_size=embedding_dim, hidden_size=hidden_size, batch_first=True)

        self.linear_layer = nn.Linear(hidden_size, class_num)

    def forward(self, x):
        batch_size, max_len = x.shape

        embedding_out = self.embedding_layer(x)

        h_0 = torch.randn(1,batch_size, self.hidden_size).to(self.device)
        _, h_n = self.rnn_layer(embedding_out,h_0)

        out = self.linear_layer(h_n).squeeze(0)
        return out


