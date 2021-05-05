from nltk.tokenize import wordpunct_tokenize
from nltk.corpus import stopwords
from gensim.parsing.porter import PorterStemmer
from gensim import corpora, models, similarities
from gensim.models import Word2Vec
import json as js
import numpy as np
import gensim.downloader as api
from load_data import *

def preprocess_document(doc):
    stopset = set(stopwords.words('english'))
    stemmer = PorterStemmer() 
    tokens = wordpunct_tokenize(doc) # split text on whitespace and punctuation
    clean = [token.lower() for token in tokens if token.lower() not in stopset and len(token) > 2] #remove stopwords
    final_doc = [stemmer.stem(word) for word in clean] #
    return clean

def preprocess_doclist(docs):
    r = []
    for d in docs:
        r.append(preprocess_document(d))
    return r

def doc2vector(doc, model):
    pdoc = preprocess_document(doc)
    v = []
    for w in pdoc:
        try: 
            wv = model.get_vector(w)
            v.append(wv)
        except:
            print('--> ', w, 'not in model')
    return v



def data2train(docsdict, querysdict, relpairs, model):
    
    data_count = len(relpairs)
    step = data_count // 10

    X, Y, XV, XY = [], [], [], []
    dataX = relpairs[step:]
    dataXV = relpairs[:step]

    for p in dataX:
        q_id, d_id, r = p
        doc = docsdict[d_id]
        query = querysdict[q_dic]
        vdoc = doc2vector(doc, model)
        vquery = doc2vector(query, model)
        x = (vquery, vdoc)
        X.append(x)
        Y.append(r)

    for p in dataXV:
        q_id, d_id, r = p
        doc = docsdict[d_id]
        query = querysdict[q_dic]
        vdoc = doc2vector(doc, model)
        vquery = doc2vector(query, model)
        x = (vquery, vdoc)
        XV.append(x)
        YV.append(r)

    print(len(X), len(Y), len(XV), len(YV))


def check_tokens_in_model(tokens):
    model = api.load('glove-wiki-gigaword-300')
    vl = []

    for w in tokens:
        try:
            wv = model.get_vector(w)
            
        except:
            print('->', w)
            vl.append(w)
    return vl

def word2id_dict(pdocs):
    dictionary = corpora.Dictionary(pdocs)
    dictionary.save('vsm.dict') # save dictionary in a vector space matrix
    return dictionary

def save_word2vect(docs, model, file_name):
    w2id_dict = word2id_dict(docs)
    wid2vect_dict = {}
    nowords = []


    for w_id, w in w2id_dict.items():
        try: 
            v = model.get_vector(w)
            wid2vect_dict[w] = [float(x) for x in v]
        except:
            print(w)
            nowords.append(w)
    
    wv = open(file_name, "w")
    js.dump(wid2vect_dict, wv)
    wv.close()
    return nowords

def save_words_info(wembedding = 'glove-wiki-gigaword-50', file_name = 'w2vect50.bin'):
    docs, docsdict = dataset_dict('../dataset/corpus/MED.ALL')
    querys, querysdict = dataset_dict('../dataset/queries/MED.QRY')

    all_docs = docs + querys
    all_docs = preprocess_doclist(all_docs)
    model = api.load(wembedding)
    save_word2vect(all_docs, model, file_name)



def main():
    save_words_info(wembedding='glove-wiki-gigaword-300', file_name='w2vect300.bin')

    fd = open('w2vect300.bin')
    d = js.load(fd)
    print(len(d))

    


if __name__ == "__main__":
    main()