#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  3 11:36:31 2018

@author: manoj
"""



import pickle
import requests
import json
from tqdm import tqdm

#%%
headers={
    "X-Mashape-Key": "Ze2iJtPsadmsh90w3Rs3YcbmbCsMp1Y9M6ojsnAAJM0QqElJan",
    "Accept": "application/json"
    }

url = "https://montanaflynn-spellcheck.p.mashape.com/check/?text={}"


def add_spellcheck(ds):
    """Input: the data source - ds
       Return: None
       Output: spell check params are added in 'response' key of the dict.
    """
    for ent in tqdm(ds):
        if ent['data_source'] == 'amt':
            q = ent['question']
            encq = '+'.join(str(q).split(" "))
            full_url = url.format(encq)
            r=requests.get(full_url, headers=headers)
            ent['response'] = json.loads(r.content)

#%%
test = pickle.load(open('/home/manoj/Downloads/counting/data/test_data_kushalformat_fixed.pkl','rb'))
train = pickle.load(open('/home/manoj/Downloads/counting/data/train_data_kushalformat_fixed.pkl','rb'))

add_spellcheck(train)
add_spellcheck(test)


#%%
c = 0
for ent in train:
    if ent.get('response',None):
        if ent['response']['corrections'] != {}:
            print (ent['question_id'],ent['response'])     
            c+=1

#%%
            
ctest = []
for ent in test:
    if ent.get('response',None):
        if ent['response']['corrections'] != {}:
            print (ent['question_id'],ent['response'])     
            ctest.append(ent)       




#%%
#release the json format files

json.dump(train,open("train_final.json","w"))
json.dump(test,open("test_final.json","w"))            
            
            
            
