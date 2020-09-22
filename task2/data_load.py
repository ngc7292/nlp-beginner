# -*- coding: utf-8 -*-
"""
__title__="data"
__author__="ngc7293"
__mtime__="2020/9/20"
"""
import pandas as pd
from tqdm import tqdm
import numpy as np
import torch

from torch.utils.data import DataLoader, Dataset
from gensim.models import KeyedVectors
from collections import Counter
from sklearn.model_selection import train_test_split


def Corpus_Extr(df):
    print('Construct Corpus...')
    corpus = []
    for i in tqdm(range(len(df))):
        corpus.append(df.Phrase[i].lower().split())
    corpus = Counter(np.hstack(corpus))
    # print(corpus)
    corpus2 = sorted(corpus, key=corpus.get, reverse=True)
    print('Convert Corpus to Integers')
    word2id = {word: idx for idx, word in enumerate(corpus2, 1)}
    word2id['<unk>'] = 0
    id2word = {idx: word for idx, word in enumerate(corpus2, 1)}
    id2word[0] = '<unk>'
    print('Convert Phrase to Integers')
    phrase2id = []
    for i in tqdm(range(len(df))):
        phrase2id.append([word2id[word] for word in df.Phrase.values[i].lower().split()])
    return id2word, word2id, phrase2id


def Pad_sequences(phrase2id, sentence_len):
    pad_sequences = np.zeros((len(phrase2id), sentence_len), dtype=int)
    for idx, row in tqdm(enumerate(phrase2id), total=len(phrase2id)):
        pad_sequences[idx, :len(row)] = np.array(row)[:sentence_len]
    return pad_sequences


class PhraseDataset(Dataset):
    def __init__(self,
                 x,
                 y=None,
                 is_train=True):
        super().__init__()
        self.x = x
        self.y = y
        self.is_train = is_train

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        if self.is_train:
            return self.x[idx], self.y[idx]
        else:
            return self.x[idx]

        # if 'Sentiment' in self.df.columns:
        #     label = self.df['Sentiment'].values[idx]
        #     item = self.pad_sequences[idx]
        #     return item, label
        # else:
        #     item = self.pad_sequences[idx]
        #     return item


def get_pretrained(id2word, model_file, vec_dim=50):
    print("loading vec embeding models...")
    model = KeyedVectors.load_word2vec_format(model_file)
    res = torch.nn.init.xavier_uniform_(torch.empty(len(id2word), vec_dim))
    count = 0
    print('convert models...')
    for idx in tqdm(id2word.keys()):
        word = id2word[idx]
        try:
            res[idx] = torch.tensor(model.get_vector(word))
            count += 1
        except:
            continue
    print("total %d words and in %s find %d words" % (len(id2word),model_file, count))
    return res.float()


def get_dataloader(batch_size=32, set_len=-1, sentence_len=50, pre_model = 'glove'):
    """
    :param batch_size:
    :return:
    """
    train = pd.read_csv('../data/train.tsv', sep='\t')
    test = pd.read_csv('../data/test.tsv', sep='\t')
    model_file = "../data/pre/glove.6B.50d.word2vec.txt"

    int2word, word2id, phrase_to_int = Corpus_Extr(train)

    pad_sequences = Pad_sequences(phrase_to_int, sentence_len)

    if pre_model == "word2vec":
        model_file = "../data/pre/word2vec.50d.word2vec.txt"
    else:
        model_file = "../data/pre/glove.6B.50d.word2vec.txt"

    embed_pretrain = get_pretrained(int2word, model_file)

    x = pad_sequences
    y = train['Sentiment'].values

    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2, random_state=42)

    if set_len != -1:
        ids = np.random.choice(len(train) - 1, set_len)
        train_set = PhraseDataset(train_x[ids], train_y[ids])
    else:
        train_set = PhraseDataset(train_x, train_y)

    test_set = PhraseDataset(test_x,test_y)
    train_loader = DataLoader(train_set, batch_size=batch_size)
    test_loader = DataLoader(test_set, batch_size=batch_size)

    return train_loader, test_loader, embed_pretrain, len(word2id)

if __name__ == '__main__':
    train_loader,test_loader,embed_pretrain, vocab_size = get_dataloader(pre_model="word2vec")
    i = next(iter(train_loader))
    print(i)
