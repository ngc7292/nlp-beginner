# -*- coding: utf-8 -*-
"""
__title__="model"
__author__="ngc7293"
__mtime__="2020/10/7"
"""
import torch.nn as nn
import torch
import torch.nn.functional as F


class lstm_char(nn.Module):
    def __init__(self, vocab_size, embedding_dim=100, hidden_dim=1024):
        super(lstm_char, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        self.embedding_layer = nn.Embedding(vocab_size, embedding_dim)
        self.lstm_layer = nn.LSTM(embedding_dim, hidden_dim, num_layers=3,batch_first=True,dropout=0, bidirectional=False)
        self.fc1 = nn.Linear(hidden_dim, 2048)
        self.fc2 = nn.Linear(2048, 4096)
        self.fc3 = nn.Linear(4096, vocab_size)

    def forward(self, input, device, hidden_state=None):
        """

        :param input: [btach_size, seq_len]
        :param device:
        :param hidden_state:
        :return:
        """
        embed_out = self.embedding_layer(input) # size is [btach_siz, seq_len, embedding_dim]
        batch_size, seq_len = input.size()
        if hidden_state is None:
            h_0 = input.data.new(3, batch_size, self.hidden_dim).fill_(0).float().to(device)
            c_0 = input.data.new(3, batch_size, self.hidden_dim).fill_(0).float().to(device)
        else:
            h_0, c_0 = hidden_state

        output, hidden_state = self.lstm_layer(embed_out, (h_0, c_0)) #size is [batch_size, seq_len, hidden_dim]
        output = torch.tanh(self.fc1(output))
        output = torch.tanh(self.fc2(output))
        output = self.fc3(output)
        output = output.reshape(batch_size * seq_len, -1)
        return output, hidden_state


if __name__ == '__main__':
    from data import Poetry

    a = Poetry()
    x, y = a.next_batch(64)
    vocab_size = len(a.word_to_id)
    device = torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')

    x = torch.from_numpy(x).long().to(device)
    y = torch.from_numpy(y).long().to(device)

    model = lstm_char(vocab_size=vocab_size).to(device)

    out,_ = model(x, device)

    c = torch.argmax(out.cpu(), axis=-1)
    print()