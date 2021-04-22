from nltk.tokenize import wordpunct_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from gensim import corpora, models, similarities


def preprocess_document(doc):
    stopset = set(stopwords.words('english'))
    stemmer = PorterStemmer() 
    tokens = wordpunct_tokenize(doc) # split text on whitespace and punctuation
    clean = [token.lower() for token in tokens if token.lower() not in stopset and len(token) > 2] #remove stopwords
    final = [stemmer.stem(word) for word in clean] #
    return final


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

