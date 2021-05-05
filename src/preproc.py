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

def doc2vector(pdoc, w2v_dict):
    v = []
    for w in pdoc:
        if w2v_dict.__contains__(w):
            v.append(w2v_dict[w])
    return v

def data2train(docsdict, queriesdict, relpairs, w2v_dict):
    
    data_count = len(relpairs)
    step = data_count // 10

    X, Y, XV, YV = [], [], [], []
    Xd, Xq , XVd, XVq = [], [], [], []
    dataX = relpairs[step:]
    dataXV = relpairs[:step]

    print(len(relpairs), len(dataX), len(dataXV))

    for p in dataX:
        q_id, d_id, r = p
        doc = preprocess_document(docsdict[d_id])
        query = preprocess_document(queriesdict[q_id])
        vdoc = doc2vector(doc, w2v_dict)
        vquery = doc2vector(query, w2v_dict)
        Xd.append(vdoc)
        Xq.append(vquery)
        # X.append(x)
        Y.append(r)

    for p in dataXV:
        q_id, d_id, r = p
        doc = preprocess_document(docsdict[d_id])
        query = preprocess_document(queriesdict[q_id])
        vdoc = doc2vector(doc, w2v_dict)
        vquery = doc2vector(query, w2v_dict)
        XVd.append(vdoc)
        XVq.append(vquery)
        # XV.append(x)
        YV.append(r)

    X = [np.array(Xd), np.array(Xq)]
    XV = [np.array(XVd), np.array(XVq)]
    Y = np.array(Y)
    YV = np.array(YV)

    return X, Y, XV, YV


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
    queries, querysdict = dataset_dict('../dataset/queries/MED.QRY')

    pdocs = preprocess_doclist(docs)
    pqueries = preprocess_doclist(queries)
    all_docs = pdocs + pqueries
    model = api.load(wembedding)
    save_word2vect(all_docs, model, file_name)



def main():
    # save_words_info(wembedding='glove-wiki-gigaword-300', file_name='w2vect300.bin')
    
    docs, docs_dict = dataset_dict('../dataset/corpus/MED.ALL')
    queries, queries_dict = dataset_dict('../dataset/queries/MED.QRY')
    relevances = read_relevances('../dataset/relevance/MED.REL')

    t, f, relpairs = conforms_pairs(relevances, len(docs_dict))
    print(len(t), len(f), len(relpairs))
    
    fd = open('w2vect50.bin')
    w2v_dict = js.load(fd)

    # d1 = docs[100]
    # print(d1)
    # print(doc2vector(d1, w2v_dict))

    X, Y, XV, YV = data2train(docs_dict, queries_dict, relpairs, w2v_dict )
    print(XV)
    print(YV)
    

    

    


if __name__ == "__main__":
    main()