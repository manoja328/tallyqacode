import spacy
import numpy as np
parser = spacy.load('en')


def getglove(sentence,Maxlen=14):    
    """ Front padding"""
    tokens = parser(sentence)

    clean_tokens = [tok.lemma_.lower().strip() for tok in tokens if
                    (not tok.is_oov and not tok.is_punct)]
    
    cln_sent = ' '.join(clean_tokens)
    par = parser(cln_sent)
    vec = []
    for tok in par:
        vec.append(tok.vector)
               
    #front padding
    if len(vec) < Maxlen:        
        padding = [np.zeros(300,dtype=np.float32)]* ( Maxlen - len(vec))
        embedding = padding + vec
    else:
        embedding = vec[0:Maxlen]
        
    embedding = np.array(embedding)
    assert len(embedding) == Maxlen
    return embedding


def getglove_old(sentence,Maxlen=14):
    """Back padding"""
    tokens = parser(sentence)

    clean_tokens = [tok.lemma_.lower().strip() for tok in tokens if
                    (not tok.is_oov and not tok.is_punct)]
    
    cln_sent = ' '.join(clean_tokens)
    par = parser(cln_sent)
    vec = []
    for tok in par:
        vec.append(tok.vector)

    #back paddding
    vec = np.array(vec)
    a = np.zeros((Maxlen,300),dtype=np.float32) # embeddding of words of sentence
    if len(vec) < Maxlen:
        a[0:len(vec)]=vec
    else:
        a = vec[0:Maxlen]
    return a


def getglove2(sentence,Maxlen=16):    
    tokens = parser(sentence.lower())    
    vec = []
    Embedding_size = 300
    for tok in tokens:
        if tok.has_vector:
            vec.append(tok.vector)
        else:
            vec.append([0]*Embedding_size)     
    vec = np.array(vec,dtype=np.float32)    
    a = np.zeros((Maxlen,Embedding_size),dtype=np.float32) # embeddding of words of sentence
    if len(vec) < Maxlen:
        a[0:len(vec)] = vec
    else:
        a = vec[0:Maxlen]
    return a    


def spacy_tokenizer(sentence,debug=False):
    '''returns all the nouns in the questions'''
    flag = ["picture","photo","image","room","floor","ground","grass"]

    for word in flag:
        if word in sentence:
            sentence = sentence.replace(word,'')

    tokens = parser(sentence)
#    tokens = [tok.lemma_.lower().strip() if tok.lemma_ != "-PRON-"
#              else tok.lower_ for tok in tokens]

    #toklist =["-JJ-", "-PRON-"]
    toklistP =["NOUN"]


    if debug==True:
        tkpos = []
        tktag = []
        tkdep = []

        for token in tokens:
    #        print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
    #              token.shape_, token.is_alpha, token.is_stop)
            #print(token.text, token.pos_,token.tag_)
            tkpos.append(token.pos_)
            tktag.append(token.tag_)
            tkdep.append(token.dep_)


        print(tkdep)
        print (tkpos)
        print (tktag)


    tokens = [tok for tok in tokens if (not tok.is_punct)]
    nouns = [tok.lemma_.lower().strip() for tok in tokens if tok.pos_  in toklistP]

    return tokens,nouns

