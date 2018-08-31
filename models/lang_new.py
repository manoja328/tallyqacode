#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  6 15:36:51 2018

@author: manoj
"""

import torch
import numpy as np
import torch.nn as nn
from .dictionary import Dictionary

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

class QuestionParser(nn.Module):
    def __init__(self,dictionaryfile = None, glove_file = None,
                 dropout=0.3, word_dim=300, ques_dim=1024 , rnn_type= 'GRU'):
        super().__init__()

        self.dropout = dropout
        self.word_dim = word_dim
        self.ques_dim = ques_dim


        dictionary = Dictionary.load_from_file(dictionaryfile)
        self.VOCAB_SIZE = dictionary.ntoken
        self.glove_file = glove_file

        self.embd = nn.Embedding(self.VOCAB_SIZE + 1, self.word_dim,
                                 padding_idx=self.VOCAB_SIZE)
        
        if rnn_type == 'GRU':
            RNN  = nn.GRU
        else:
            RNN = nn.LSTM
        
        self.rnn = RNN(self.word_dim, self.ques_dim)
        self.drop = nn.Dropout(self.dropout)
        self.glove_init()

    def glove_init(self):
        print("initialising with glove embeddings")
        glove_embds = torch.from_numpy(np.load(self.glove_file))
        assert glove_embds.size() == (self.VOCAB_SIZE, self.word_dim)
        self.embd.weight.data[:self.VOCAB_SIZE] = glove_embds
        print("done..")

    def forward(self, questions):
        # (B, MAXLEN)
        # print("question size ", questions.size())
        questions = questions.t()  # (MAXLEN, B)
        questions = self.embd(questions)  # (MAXLEN, B, word_size)
        _, (q_emb) = self.rnn(questions)
        q_emb = q_emb[-1]  # (B, ques_size)
        q_emb = self.drop(q_emb)
        return q_emb


#%%
class ScoringFunction(nn.Module):

    def __init__(self, ques_dim, dropout=0.3, v_dim=2048, score_dim=1024):
        super(ScoringFunction, self).__init__()

        self.q_dim = ques_dim
        self.dropout = dropout
        self.v_dim = v_dim
        self.score_dim = score_dim

        self.v_drop = nn.Dropout(self.dropout)
        self.q_drop = nn.Dropout(self.dropout)
        self.v_proj = FCNet([self.v_dim, self.score_dim])
        self.q_proj = FCNet([self.q_dim, self.score_dim])
        self.s_drop = nn.Dropout(self.dropout)

    def forward(self, v, q):
        """
        v: [batch, k, vdim]
        q: [batch, qdim]
        """
        batch, k, _ = v.size()
        v = self.v_drop(v)
        q = self.q_drop(q)
        v_proj = self.v_proj(v)  # [batch, k, qdim]
        q_proj = self.q_proj(q).unsqueeze(1).repeat(1, k, 1)  # [batch, k, qdim]
        s = v_proj * q_proj
        s = self.s_drop(s)
        return s  # (B, k, score_dim)


class GTUScoringFunction(nn.Module):

    def __init__(self, ques_dim, dropout=0.3, v_dim=2048, score_dim=2048):
        super(GTUScoringFunction, self).__init__()

        self.q_dim = ques_dim
        self.dropout = dropout
        self.v_dim = v_dim
        self.score_dim = score_dim

        self.predrop = nn.Dropout(self.dropout)
        self.dense1 = weight_norm(nn.Linear(self.v_dim + self.q_dim, self.score_dim), dim=None)
        self.dense2 = weight_norm(nn.Linear(self.v_dim + self.q_dim, self.score_dim), dim=None)

        self.s_drop = nn.Dropout(self.dropout)

    def forward(self, v, q):
        """
        v: [batch, k, vdim]
        q: [batch, qdim]
        """
        batch, k, _ = v.size()

        q = q[:, None, :].repeat(1, k, 1)  # (B, k, q_dim)
        vq = torch.cat([v, q], dim=2)  # (B, k, v_dim + q_dim)

        vq = self.predrop(vq)

        y = F.tanh(self.dense1(vq))  # (B, k, score_dim)
        g = F.sigmoid(self.dense2(vq))  # (B, k, score_dim)

        s = y * g
        s = self.s_drop(s)
        return s  # (B, k, score_dim)



#%%
    
if __name__ == '__main__':       
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
