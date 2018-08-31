#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 28 08:32:02 2018

@author: manojacharya
"""

import json
import numpy as np
import os
import pickle
import config
#%%
def loadjs(file):
    with open(file) as f:
        js = json.load(f)
        return js 
    print ("file not found")

def loadpickle(file):
    with open(file,'rb') as f:
        js = pickle.load(f)
        return js 
    print ("file not found")

def getfile(folder):
    file = os.listdir(folder)[0]
    return os.path.join(folder,file)

def qidtoentry(ds):
    entries = {}
    for ent in ds:
        qid = int(ent['question_id'])
        entries[qid] = ent
    return entries
def getcondition(ds):
    c = []
    for ent in ds:
        if ent['data_source'] == 'amt':
            if 'letter' in ent['question']:
                c.append(ent)
    return c



#%%
ds = 'Ourdb'
D = '/home/manoj/mutan'

testset = loadpickle(config.dataset['Ourdb']['test'])


file = getfile(os.path.join(D,ds,'updown'))
updown = loadjs(file)
updown_qent = qidtoentry(updown)

RCN = np.load("/home/manoj/Videos/nn/utils/ads.npy")
pred_reg_clip = RCN.tolist()
testqids = [ ent['question_id'] for ent in testset]
RCN_qent = dict(zip(testqids , pred_reg_clip))
#%%
label = getcondition(testset)

gt  = [ ent['answer'] for ent in label]
updown_pred = [  updown_qent[ent['question_id']]['answer'] for ent in label]
import utils
utils.accuracy(gt,updown_pred)
RCN_pred = [  RCN_qent[ent['question_id']] for ent in label]
utils.accuracy(gt,RCN_pred)


#%%
from tqdm import tqdm
import utils
import os
import matplotlib.pyplot as plt
#%%
entdis = []
for ent in tqdm(testset):
    qid = ent['question_id'] 
    if RCN_qent[qid] !=  int(updown_qent[qid]['answer']):
        if  RCN_qent[qid] != ent['answer'] and ent['data_source'] == 'amt':
            if int(updown_qent[qid]['answer']) == ent['answer']:
                ent['disagree'] = 'RCN = {} updown = {}'.format(RCN_qent[qid],int(updown_qent[qid]['answer']))
                entdis.append(ent)
                path = os.path.join('/home/manoj/',ent['image'])
                a = utils.pil_loader(path)
                plt.imshow(np.asarray(a))
                plt.title("{} {}".format(ent['question'],ent['answer']))
                plt.xlabel(ent['disagree'])
                plt.ylabel(ent['question_id'])
                l = ent['image'].split("/")[-1]
                plt.savefig("disagg/"+l,dpi=150)
                plt.close()


