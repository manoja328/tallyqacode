import numpy as np
import spacy
import pandas as pd
import os
parser = spacy.load('en')
from nltk.parse.stanford import StanfordParser
x = np.loadtxt(os.path.join('/home/manoj/Downloads/counting/','coco.names'),delimiter='\n',dtype=str)
coco= sorted(x.tolist())
import regex as re

#%%


import regex as re

cocopl = [ 'couches',
        'aeroplanes',
        'airplanes',
 'apples',
 'backpacks',
 'bananas',
 'baseball bats',
 'baseball gloves',
 'bears',
 'beds',
 'benches',
 'bicycles',
 'birds',
 'boats',
 'books',
 'bottles',
 'bowls',
 'broccolis',
 'buses',
 'cakes',
 'cars',
 'carrots',
 'cats',
 'cell phones',
 'chairs',
 'clocks',
 'cows',
 'cups',
 'dining tables',
 'dogs',
 'donuts',
 'elephants',
 'fire hydrants',
 'forks',
 'frisbees',
 'giraffes',
 'hair driers',
 'handbags',
 'horses',
 'hot dogs',
 'keyboards',
 'kites',
 'knives',
 'laptops',
 'microwaves',
 'motorbikes',
 'motorcycles',
 'mice',
 'oranges',
 'ovens',
 'parking meters',
 'people',
 'pizzas',
 'potted plants',
 'refrigerators',
 'remotes',
 'sandwiches',
 'scissors',
 'sheep',
 'sinks',
 'skateboards',
 'ski',
 'snowboards',
 'sofas',
 'spoons',
 'sports balls',
 'stop signs',
 'suitcases',
 'surfboards',
 'teddy bears',
 'tennis rackets',
 'ties',
 'toasters',
 'toilets',
 'toothbrushes',
 'traffic lights',
 'trains',
 'trucks',
 'tvs',
 'umbrellas',
 'vases',
 'wine glasses',
 'zebras']


cocos = ['couch',
 'aeroplane',
 'airplane',
 'apple',
 'backpack',
 'banana',
 'baseball bat',
 'baseball glove',
 'bear',
 'bed',
 'bench',
 'bicycle',
 'bird',
 'boat',
 'book',
 'bottle',
 'bowl',
 'broccoli',
 'bus',
 'cake',
 'car',
 'carrot',
 'cat',
 'cell phone',
 'chair',
 'clock',
 'cow',
 'cup',
 'dining table',
 'dog',
 'donut',
 'elephant',
 'fire hydrant',
 'fork',
 'frisbee',
 'giraffe',
 'hair drier',
 'handbag',
 'horse',
 'hot dog',
 'keyboard',
 'kite',
 'knife',
 'laptop',
 'microwave',
 'motorbike',
 'motorcycle',
 'mouse',
 'orange',
 'oven',
 'parking meter',
 'person',
 'pizza',
 'potted plant',
 'refrigerator',
 'remote',
 'sandwich',
 'scissors',
 'sheep',
 'sink',
 'skateboard',
 'snowboard',
 'sofa',
 'spoon',
 'sports ball',
 'stop sign',
 'suitcase',
 'surfboard',
 'teddy bear',
 'tennis racket',
 'tie',
 'toaster',
 'toilet',
 'toothbrush',
 'traffic light',
 'train',
 'truck',
 'tv',
 'umbrella',
 'vase',
 'wine glass',
 'zebra']

#%%

def match(x):
    template = ["are in the picture","are in the photo",
                "can be seen","can you see","are there","are visible","do you see"]


    pattern1 = re.compile('('+ '|'.join(cocopl) +') '+ '('+ '|'.join(template) +')')
    pattern2 = re.compile('('+ '|'.join(cocos) +') '+ '('+ '|'.join(template) +')')
    #x= "How many dog can you see?"
    match1  = pattern1.match(x)
    match2  = pattern2.match(x)
    #print (re.findall(r"How many"+ '|'.join(coco) + '|'.join(template)+'?',x))


    if match2 is not None: # it got someting out of regex match
        #print('HERE')
        return True # Simple

    if match1 is not None: # it got someting out of regex match
        #print('HERE')
        return True # Simple


    else:
        return False # complex


#for q in gettrain1['questions']:
#    if not match(q):
#        print (q)
#
#for q in gettest1['questions']:
#    if not match(q):
#        print (q)
#

#for i,que in enumerate(dfsel.question):
#    if match(que):
#        print (i,que)


def issimple(sentence,debug=False):
    sentence = sentence.replace('How many ','')
    doc = parser(sentence)
    #p = list(doc.noun_chunks)
    #jjcn = 0
    simples = True
    complexs = False
    #print (p)

    tkpos = []
    tktag = []
    tkdep = []
    for token in doc:
#        print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
#              token.shape_, token.is_alpha, token.is_stop)
        #print(token.text, token.pos_,token.tag_)
        tkpos.append(token.pos_)
        tktag.append(token.tag_)
        tkdep.append(token.dep_)

    if debug==True:
        print(tkdep)
#        print ( tkpos)
#        print ( tktag)

    if match(sentence):
        return simples
    else:
        for token in doc:
            if token.dep_ in ['amod','aux','acomp','pobj','prep','dobj','compound','ccomp']:
                return complexs

        return simples

#    if 'IN' in tktag and 'in' not in sentence:
#        return complexs
#    if tktag.count('NNS') >=2:
#        return complexs

#    if token.tag_ =='NOUN' or token.tag_ =='VERB':
#        print (token)
#        jjcn +=1
#    #print (p)
#    print (tkl)
#    if len(p) <= 1 and jjcn<2:
#        return simples
#    return complexs

#%%
sentences = ['How many dogs are there?',
             'How many dogs are in the picture?',
             'How many dogs can be seen?',
             'How many dogs are visible?',
             'How many people are wearing a red shirt?',
             'How many people can cross the street?',
             'How many giraffes are standing up?',
             'How many red dogs are there?',
             'How many red dogs are visible?',
             'How many red dogs can be seen?',
             'How many chairs are in front of the tv?',
             'How many mugs are on the counter?',
             'How many dogs are to the left of the house?',
             'How many dogs are red?',
             'How many dogs are standing?',
             'How many baby lions are there?']
for sentence in sentences:
    if issimple(sentence,True):
        print (sentence,"------> Simple")
    else:
        print (sentence,"------> Complex")

#%%
#jjcn = 0
#doc = parser(sentence)
#for token in doc:
#    print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
#          token.shape_, token.is_alpha, token.is_stop)
#    if token.tag_ =='JJ':
#        jjcn +=1



#%%

#column_names = ["HITId","WorkerId","WorkTimeInSeconds","Answer.answer","Answer.question",\
#                    "Answer.url","AssignmentStatus"]
#csvfile = '/home/manoj/Downloads/Batch_final.csv'
#df = pd.read_csv(csvfile,header = 0)
#dfsel = df[column_names]
#dfsel.columns = ["hitid","workerid","t","answer","question","image","assnmentstatus"]

#%%
#c =0
#for i,que in enumerate(dfsel.question):
#    if issimple(que):
#        print (i,que)
#        c+=1
#print (c,100*c/21405,'%')



#%%
#`113

name= 'HowmanyQA'
settings = {}
import pickle

def getcountstats(pklfile):
    stest=0
    vgtest=0
    if isinstance(pklfile,str):
        pk = pickle.load(open(pklfile,'rb'))
    else:
        pk = pklfile
    N = len(pk['questions'])
    for i,q in enumerate(pk['questions']):
        if not issimple(q):
            if 'VG' in pk['images'][i]:
                vgtest+=1
            stest +=1
    print (N, stest, N-stest)
    return stest,vgtest


#strain,vgtr = getcountstats('/home/manoj/Downloads/HowMany-QA/howmanyQA_train.pkl')
#sdev , vgdev= getcountstats('/home/manoj/Downloads/HowMany-QA/howmanyQA_dev.pkl')
#stest,vgtest = getcountstats('/home/manoj/Downloads/HowMany-QA/howmanyQA_test.pkl')

#%%

#strain,vgtr= getcountstats('/home/manoj/VQA_our_train.pkl')
#stest, vgtest= getcountstats('/home/manoj/VQA_our_test.pkl')

#%%

#strain,vgtr= getcountstats('/home/manoj/counting/TDIUC/traindata.pkl')
#stest, vgtest= getcountstats('/home/manoj/counting/TDIUC/testdata.pkl')


