from nltk.tokenize import wordpunct_tokenize
from nltk.corpus import stopwords
from gensim.parsing.porter import PorterStemmer
from gensim import corpora, models, similarities
from gensim.models import Word2Vec
import json as js
import numpy as np
import gensim.downloader as api
# from load_data import *

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
    # v = ave_sentence(v, 50)
    return np.array(v)

def ave_sentence(doc_emb, dim):
    """
    Return the average between the words vector embedding in a sentences 
    """
    dv = []
    if (len(doc_emb) > 0):
        dv = doc_emb[0]
        for i in range(1, len(doc_emb)):
            for j in range(dim):
                dv[j] += doc_emb[i][j]
            dv[j] = float(dv[j])/float(dim)
    return dv 

def data2train(docsdict, queriesdict, relpairs, w2v_dict):
    
    data_count = len(relpairs)
    step = data_count // 5

    X, Y, XV, YV = [], [], [], []
    dataX = relpairs[step:]
    dataXV = relpairs[:step]

    for p in dataX:
        q_id, d_id, r = p
        doc = docsdict[d_id]
        query = queriesdict[q_id]
        vdoc = doc2vector(doc, w2v_dict)
        vquery = doc2vector(query, w2v_dict)
        if(len(vdoc) == 0 or len(vquery) == 0):
            print("-> empty doc")

        x = (vdoc, vquery)
        X.append(x)
        Y.append(r)

    for p in dataXV:
        q_id, d_id, r = p
        doc = docsdict[d_id]
        query = queriesdict[q_id]
        vdoc = doc2vector(doc, w2v_dict)
        vquery = doc2vector(query, w2v_dict)
        if(len(vdoc) == 0 or len(vquery) == 0):
            print("-> empty doc")
        x = (vdoc, vquery)
        XV.append(x)
        YV.append(r)


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
    dictionary.save('./word2vect/vsm.dict') # save dictionary in a vector space matrix
    return dictionary

def doc2bows(dictionary, doc):
    return dictionary.doc2bow(doc) 

def list_docs2bows(dictionary, docs):
    corpus = [dictionary.doc2bow(doc) for doc in docs]
    corpora.MmCorpus.serialize('./word2vect/vsm_docs.mm', corpus) #Serialize the corpus using the Matrix Market format
    return corpus


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


##############################################################################################
# MATCH-PYRAMID PREPROC 
##############################################################################################

def data2train_mp(docsdict, queriesdict, relpairs, w2v_dict, query_len, doc_len, emb_dim):
    
    data_count = len(relpairs)
    step = data_count // 5

    X_query, X_doc, Y, XV_query, XV_doc, YV = [], [], [], [], [], []
    dataX = relpairs[step:]
    dataXV = relpairs[:step]

    for p in dataX:
        q_id, d_id, r = p
        doc = docsdict[d_id]
        query = queriesdict[q_id]
        vdoc = doc2vector_mp(doc, w2v_dict, doc_len, emb_dim)
        vquery = doc2vector_mp(query, w2v_dict, query_len, emb_dim)

        X_query.append(vquery)
        X_doc.append(vdoc)

        c = [0, 0]
        c[r] = 1
        Y.append(c)

    for p in dataXV:
        q_id, d_id, r = p
        doc = docsdict[d_id]
        query = queriesdict[q_id]
        vdoc = doc2vector_mp(doc, w2v_dict, doc_len, emb_dim)
        vquery = doc2vector_mp(query, w2v_dict, query_len, emb_dim)
        XV_query.append(vquery)
        XV_doc.append(vdoc)
        c = [0, 0]
        c[r] = 1
        YV.append(c)

    return np.array(X_query), np.array(X_doc), np.array(Y), np.array(XV_query), np.array(XV_doc), np.array(YV)


def doc2vector_mp(pdoc, w2v_dict, length, emb_dim):
    pad = [0] * emb_dim
    v = [pad] * length
    for i,w in enumerate(pdoc):
        if i == length:
            break
        if w2v_dict.__contains__(w):
            v[i] = w2v_dict[w]
    # v = ave_sentence(v, 50)
    return np.array(v)


def main():
    print('preprocessing info')
    # save_words_info(wembedding='glove-wiki-gigaword-50', file_name='ciri50.bin')
  
    


if __name__ == "__main__":
    main()