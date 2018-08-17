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


SPECIAL_TOKENS = {
  '<NULL>': 0,
  '<START>': 1,
  '<END>': 2,
  '<UNK>': 3,
}

def tokenize(s, delim=' ',
      add_start_token=True, add_end_token=True,
      punct_to_keep=None, punct_to_remove=None):
    """
    Tokenize a sequence, converting a string s into a list of (string) tokens by
    splitting on the specified delimiter. Optionally keep or remove certain
    punctuation marks and add start and end tokens.
    """

    s = s.replace("n't"," not")
    
    if punct_to_keep is not None:
        for p in punct_to_keep:
            s = s.replace(p, '%s%s' % (delim, p))

    if punct_to_remove is not None:
        for p in punct_to_remove:
            s = s.replace(p, '')

    s = " ".join(s.split()) # remove multiple contiguous whitespace
    tokens = s.split(delim)
    if add_start_token:
        tokens.insert(0, '<START>')
    if add_end_token:
        tokens.append('<END>')
    return tokens


def build_vocab(sequences, min_token_count=1, delim=' ',
      punct_to_keep=None, punct_to_remove=None):

    token_to_count = {}
    tokenize_kwargs = {
            'delim': delim,
            'punct_to_keep': punct_to_keep,
            'punct_to_remove': punct_to_remove,
    }
    for seq in sequences:
        seq_tokens = tokenize(seq, **tokenize_kwargs,
                        add_start_token=False, add_end_token=False)
        for token in seq_tokens:
            if token not in token_to_count:
                token_to_count[token] = 0
            token_to_count[token] += 1

    token_to_idx = {}
    for token, idx in SPECIAL_TOKENS.items():
        token_to_idx[token] = idx
    for token, count in sorted(token_to_count.items()):
        if count >= min_token_count:
            token_to_idx[token] = len(token_to_idx)

    return token_to_idx


def encode(tokens, token_to_idx, allow_unk=False):
    seq_idx = []
    for token in tokens:
        if token not in token_to_idx:
            if allow_unk:
                token = '<UNK>'
            else:
                raise KeyError('Token "%s" not in vocab' % token)
        seq_idx.append(token_to_idx[token])
    return seq_idx

def encode_sent(sentence , tkidx , length = None):
    seq_tokens = tokenize(sentence, delim =' ',punct_to_remove =['?'],
                          add_start_token=False, add_end_token=False)
    idxs = encode(seq_tokens,tkidx,allow_unk=True)
    if length is not None:
        pad = [SPECIAL_TOKENS['<END>']]*(length - len(idxs))
        idxs.extend(pad)
    return idxs



