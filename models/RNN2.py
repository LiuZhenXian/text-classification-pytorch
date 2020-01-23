# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Config(object):

    def __init__(self, dataset, embedding):
        self.model_name = 'RNN2'
        self.train_path = dataset + '/data/train.txt'
        self.dev_path = dataset + '/data/dev.txt'
        self.test_path = dataset + '/data/test.txt'
        self.class_list = [x.strip() for x in open(
            dataset + '/data/class.txt').readlines()]
        self.vocab_path = dataset + '/data/vocab.pkl'
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'
        self.embedding_pretrained = torch.tensor(
            np.load(dataset + '/data/' + embedding)["embeddings"].astype('float32'))\
            if embedding != 'random' else None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.log_path = dataset + '/log/' + self.model_name
        self.dropout = 0.5
        self.require_improvement = 1000
        self.num_classes = len(self.class_list)
        self.n_vocab = 0
        self.num_epochs = 15
        self.batch_size = 128
        self.pad_size = 50
        self.learning_rate = 0.91*1e-3
        self.embed = self.embedding_pretrained.size(1)\
            if self.embedding_pretrained is not None else 300
        self.hidden_size = 128
        self.num_layers = 1



class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        if config.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
        else:
            self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=config.n_vocab - 1)
        self.lstm = nn.LSTM(config.embed, config.hidden_size, config.num_layers,
                            bidirectional=True, batch_first=True, dropout=config.dropout)
        self.maxpool = nn.MaxPool1d(config.pad_size)
        self.fc = nn.Linear(config.hidden_size * 2 + config.embed, config.num_classes)
        self.gru =nn.GRU(300,config.hidden_size, config.num_layers,
                                   batch_first=True,bidirectional=True,dropout=config.dropout)
        self.dropout = nn.Dropout(config.dropout)
        self.m = nn.Softmax(dim=1)

    def forward(self, x):
        x, _ = x
        embed = self.embedding(x)
        out, _ = self.gru(embed)
        out = torch.cat((embed, out), 2)
        out = F.relu(out)
        out = out.permute(0, 2, 1)
        out = self.maxpool(out).squeeze()
        out = self.dropout(out)
        out = self.fc(out)

        return out
