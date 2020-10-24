# -*- coding: utf-8 -*-
"""
__title__="data"
__author__="ngc7293"
__mtime__="2020/9/24"
"""
from torchtext.data import Iterator, BucketIterator
from torchtext import data
import torchtext
import torch


def get_iter(data_path, device, batch_size):
    TEXT = data.Field(batch_first=True, include_lengths=True, lower=True)
    LABEL = data.Field(batch_first=True)

    fields = {'sentence1': ('x0', TEXT),
              'sentence2': ('x1', TEXT),
              'gold_label': ('y', LABEL)
              }

    train_data, dev_data, test_data = data.TabularDataset.splits(
        path=data_path,
        train='snli_1.0_train.jsonl',
        validation='snli_1.0_dev.jsonl',
        test='snli_1.0_test.jsonl',
        format='json',
        fields=fields,
        filter_pred=lambda ex: ex.y != '-'  # filter the example which label is '-'(means unlabeled)
    )

    vector = torchtext.vocab.GloVe()

    TEXT.build_vocab(train_data, vectors=vector, unk_init=torch.Tensor.normal_)

    LABEL.build_vocab(dev_data)

    train_iter, dev_iter = BucketIterator.splits(
        (train_data, dev_data),
        batch_size=batch_size,
        device=device,
        sort_key=lambda x: len(x.x0) + len(x.x1),
        sort_within_batch=True,
        repeat=False,
        shuffle=True
    )

    test_iter = Iterator(
        test_data,
        batch_size=batch_size,
        device=device,
        sort=False,
        sort_within_batch=False,
        repeat=False,
        shuffle=False)

    return train_iter, dev_iter, test_iter, TEXT, LABEL


if __name__ == '__main__':
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    data_path = "../data/snli/snli_1.0/"
    train_iter, dev_iter, test_iter, _, _ = get_iter(data_path, device, 32)

    for i in train_iter:
        print(i.x0)
        print(i.x1)
        print(i.y)
        break
