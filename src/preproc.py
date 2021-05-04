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

def check_tokens_in_model(tokens):
    model = api.load('glove-wiki-gigaword-50')
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
    docs = read_dataset('../dataset/corpus/MED.ALL') 
    d, pdocs = word2id_dict(docs)
    l = check_tokens_in_model(d.values())
    print(len(d), len(l))

if __name__ == "__main__":
    main()