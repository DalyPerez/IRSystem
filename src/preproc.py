from nltk.tokenize import wordpunct_tokenize
from nltk.corpus import stopwords
from gensim.parsing.porter import PorterStemmer
from gensim import corpora, models, similarities
from gensim.models import Word2Vec
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

def word2id_dict(docs):
    pdocs = [preprocess_document(doc) for doc in docs]
    dictionary = corpora.Dictionary(pdocs)
    dictionary.save('vsm.dict') # save dictionary in a vector space matrix
    return dictionary ,pdocs

def doc2bows(dictionary, doc):
    vdoc = preprocess_document(doc)
    return dictionary.doc2bow(vdoc) 

def list_docs2bows(dictionary, docs):
    corpus = [dictionary.doc2bow(doc) for doc in docs]
    corpora.MmCorpus.serialize('vsm_docs.mm', corpus) #Serialize the corpus using the Matrix Market format
    return corpus

def main():
    docs = dataset_dict('../dataset/corpus/MED.ALL')
    querys = dataset_dict('../dataset/queries/MED.QRY')
    relevances = read_relevances('../dataset/relevance/MED.REL')
    pairs = conforms_pairs(relevances, len(docs))

    model = api.load('glove-wiki-gigaword-300')

    data2train(docs, querys, pairs, model)

if __name__ == "__main__":
    main()