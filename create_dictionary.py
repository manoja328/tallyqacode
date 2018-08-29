#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 29 10:30:29 2018

@author: manojacharya
"""
import numpy as np
from config import dataset
import pickle
from models.dictionary import Dictionary

def create_dictionary(ds):
    dictionary = Dictionary()
    entries = []   
    for group in ['train','test']:        
        with open( dataset[ds][group],'rb') as f:
            d = pickle.load(f)
            entries.extend(d)
    for ent in entries:
        qs = ent['question']
        dictionary.tokenize(qs, True)
    return dictionary


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
        vals = [float(val) for val in  vals[1:]]
        word2emb[word] = np.array(vals)
    for idx, word in enumerate(idx2word):
        if word not in word2emb:
            continue
        weights[idx] = word2emb[word]
    return weights, word2emb


if __name__ == '__main__':
    ds = 'Ourdb'
    d = create_dictionary(ds)
    d.dump_to_file('data/dictionary.pickle')

    d = Dictionary.load_from_file('data/dictionary.pickle')
    emb_dim = 300
    glove_file = 'data/glove/glove.6B.%dd.txt' % emb_dim
    weights, word2emb = create_glove_embedding_init(d.idx2word, glove_file)
    np.save('data/glove6b_init_%dd.npy' % emb_dim, weights)
