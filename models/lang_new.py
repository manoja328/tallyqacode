#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  6 15:36:51 2018

@author: manoj
"""

import json
import torch
from torch.autograd import Variable
import os
import numpy as np
import torch.nn as nn
import pickle
#%%

class Dictionary(object):
    def __init__(self, word2idx=None, idx2word=None):
        if word2idx is None:
            word2idx = {}
        if idx2word is None:
            idx2word = []
        self.word2idx = word2idx
        self.idx2word = idx2word

    @property
    def ntoken(self):
        return len(self.word2idx)

    @property
    def padding_idx(self):
        return len(self.word2idx)

    def tokenize(self, sentence, add_word=None):
        sentence = sentence.lower()
        sentence = sentence.replace(',', '').replace('?', '').replace('\'s', ' \'s')
        words = sentence.split()
        tokens = []
        if add_word:
            for w in words:
                tokens.append(self.add_word(w))
        else:
            for w in words:
                tokens.append(self.word2idx[w])
        return tokens

    def dump_to_file(self, path):
        pickle.dump([self.word2idx, self.idx2word], open(path, 'wb'))
        print('dictionary dumped to %s' % path)

    @classmethod
    def load_from_file(cls, path):
        print('loading dictionary from %s' % path)
        word2idx, idx2word = pickle.load(open(path, 'rb'))
        d = cls(word2idx, idx2word)
        return d

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)



def create_dictionary(dataroot):
    dictionary = Dictionary()
    files = [
        'train_final.json',
        'test_final.json'
    ]
    for path in files:
        question_path = os.path.join(dataroot, path)
        qs = json.load(open(question_path))
        for q in qs:
            dictionary.tokenize(q['question'], True)
    return dictionary



#%%

class QuestionEmbedding(nn.Module):
    def __init__(self, in_dim, num_hid, nlayers, bidirect, dropout, rnn_type='GRU'):
        """Module for question embedding
        """
        super(QuestionEmbedding, self).__init__()
        assert rnn_type == 'LSTM' or rnn_type == 'GRU'
        rnn_cls = nn.LSTM if rnn_type == 'LSTM' else nn.GRU

        self.rnn = rnn_cls(
            in_dim, num_hid, nlayers,
            bidirectional=bidirect,
            dropout=dropout,
            batch_first=True)

        self.in_dim = in_dim
        self.num_hid = num_hid
        self.nlayers = nlayers
        self.rnn_type = rnn_type
        self.ndirections = 1 + int(bidirect)

    def init_hidden(self, batch):
        # just to get the type of tensor
        weight = next(self.parameters()).data
        hid_shape = (self.nlayers * self.ndirections, batch, self.num_hid)
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(*hid_shape).zero_()),
                    Variable(weight.new(*hid_shape).zero_()))
        else:
            return Variable(weight.new(*hid_shape).zero_())

    def forward(self, x):
        # x: [batch, sequence, in_dim]
        batch = x.size(0)
        hidden = self.init_hidden(batch)
        self.rnn.flatten_parameters()
        output, hidden = self.rnn(x, hidden)

        if self.ndirections == 1:
            return output[:, -1]

        forward_ = output[:, -1, :self.num_hid]
        backward = output[:, 0, self.num_hid:]
        return torch.cat((forward_, backward), dim=1)

    def forward_all(self, x):
        # x: [batch, sequence, in_dim]
        batch = x.size(0)
        hidden = self.init_hidden(batch)
        self.rnn.flatten_parameters()
        output, hidden = self.rnn(x, hidden)
        return output


class WordEmbedding(nn.Module):
    """Word Embedding

    The ntoken-th dim is used for padding_idx, which agrees *implicitly*
    with the definition in Dictionary.
    """
    def __init__(self, ntoken, emb_dim, dropout):
        super(WordEmbedding, self).__init__()
        self.emb = nn.Embedding(ntoken+1, emb_dim, padding_idx=ntoken)
        self.dropout = nn.Dropout(dropout)
        self.ntoken = ntoken
        self.emb_dim = emb_dim

    def init_embedding(self, np_file):
        weight_init = torch.from_numpy(np.load(np_file))
        assert weight_init.shape == (self.ntoken, self.emb_dim)
        self.emb.weight.data[:self.ntoken] = weight_init

    def forward(self, x):
        emb = self.emb(x)
        emb = self.dropout(emb)
        return emb


#%%

def create_glove_embedding_init(idx2word, glove_file):
    word2emb = {}
    with open(glove_file, 'r') as f:
        entries = f.readlines()
    emb_dim = len(entries[0].split(' ')) - 1
    print('embedding dim is %d' % emb_dim)
    weights = np.zeros((len(idx2word), emb_dim), dtype=np.float32)

    for entry in entries:
        vals = entry.split(' ')
        word = vals[0]
        vals = [float(vals[i]) for i in range(1,len(vals))]
        word2emb[word] = np.array(vals)
    for idx, word in enumerate(idx2word):
        if word not in word2emb:
            continue
        weights[idx] = word2emb[word]
    return weights, word2emb


def tokenize_ques(dictionary,question,max_length=14):
    """Tokenizes the questions.

    This will add q_token in each entry of the dataset.
    -1 represent nil, and should be treated as padding_idx in embedding
    """

    tokens = dictionary.tokenize(question, add_word = None)
    tokens = tokens[:max_length]
    if len(tokens) < max_length:
        # Note here we pad in front of the sentence
        padding = [dictionary.padding_idx] * (max_length - len(tokens))
        tokens = padding + tokens
    assert len(tokens) ==  max_length , "Tokens NOT EQUAL TO MAX length."
    return np.array(tokens)


#%%
    
if __name__ == '__main__':
    d = create_dictionary(dataroot = '/home/manoj/best_training_ever2')
    d.dump_to_file('../data/dictionary.pkl')
    d = Dictionary.load_from_file('../data/dictionary.pkl')
    emb_dim = 300
    glove_file = '../data/glove/glove.6B.{}d.txt'.format(emb_dim)
    weights, word2emb = create_glove_embedding_init(d.idx2word, glove_file)
    np.save('../data/glove6b_init_{}d.npy'.format(emb_dim), weights)
        
    tokens = tokenize_ques(d,"How many dogs?")
    question = torch.from_numpy(tokens)
    question = Variable(question).unsqueeze(0)    
    w_emb = WordEmbedding(d.ntoken, emb_dim =emb_dim, dropout=0.0)
    w_emb.init_embedding('../data/glove6b_init_{}d.npy'.format(emb_dim))
    num_hid = 512
    q_emb = QuestionEmbedding( in_dim = emb_dim, num_hid = num_hid,\
                              nlayers = 1, bidirect = False, dropout = 0.0,
                              rnn_type= 'LSTM' )    
    w_emb = w_emb(question)
    q_enc = q_emb(w_emb) # [batch, q_dim]
