import re
import sys
import os
import csv
import json as js
import numpy as np
import gensim.downloader as api
from preproc import *

def read_dataset(user_path): 
    """
        Return a documents list of a specific path
    """
    path=user_path[:-8]  # Erase the file name and keep the path
    if os.path.exists(path): # the user has provided a file path with a set of texts
       try:
           list_texts = re.split(".I \d*\n.W\n",open(user_path).read())[1:] # Split text file with the delimiter, erase first delimiter
           return list_texts
       except IOError:
            print (user_path + " - No such file or directory")
            sys.exit(0)
    else: 
       return user_path

def dataset_dict(user_path):
    """
    Return a {doc_id -> doc text} dictionary
    """
    docs = read_dataset(user_path)
    pdocs = []
    id2doc = {}
    count = 1
    for d in docs:
        pdoc = preprocess_document(d)
        pdocs.append(pdoc)
        id2doc[count] = pdoc
        count = count + 1
    return pdocs, id2doc

def read_relevances(rel_path):
    """
    Return a { query_id -> [relevances docs id] } dictionary
    """
    lines = re.split("\n",open(rel_path).read())
    lines = [re.split(' 0 | 1', l) for l in lines]  
    lines.remove([''])
    relevances= {}
    for l in lines:
        q_id = int(l[0])
        if not relevances.__contains__(q_id): 
            relevances[q_id] = []
        relevances[q_id].append(int(l[1]))
    return relevances

def conforms_pairs(query_relevances, total_docs):
    """
        Conforms (q_id, d_id, relevance) pairs
        true_pairs - (q_id, d_id, 1)
        false_pairs - (q_id, d_id, 0)
    """
    total = len(query_relevances)
    all_pairs = []
    true_pairs = []
    false_pairs = []

    for i in range(1, total + 1):
        if (len(query_relevances[1]) == 0):
            continue
        for d in range(1, total_docs + 1):
            if d in query_relevances[i]:
                true_pairs.append((i, d, 1)) # (query, doc, rel)
            else:
                false_pairs.append((i, d, 0))
   
    return true_pairs, false_pairs, true_pairs + false_pairs

# for dataset in json format
def read_all(json_path):
    wd = open(json_path)
    corpus = js.load(wd)
    corpus_proc = {}
    pdocs = []

    for k, info in corpus.items():
        if(info.__contains__('text')):
            doc = info['title'] + info['text']
        else: 
            doc = info['abstract']

        pdoc = preprocess_document(doc)
        corpus_proc[int(k)] = pdoc
        pdocs.append(pdoc)
    return corpus_proc, pdocs

def read_qry(json_path):
    wd = open(json_path)
    queries = js.load(wd)
    queries_proc = {}
    pqueries = []

    for k, info in queries.items():
        # print(k)
        pquery = preprocess_document(info['text'])
        queries_proc[int(k)] = pquery
        pqueries.append(pquery)
    return queries_proc, pqueries

def read_rel(json_path, total_queries):
    wd = open(json_path)
    rel_dict = js.load(wd)
    # print("---------->", len(rel_dict))
    relevances = {}

    for q_id, info in rel_dict.items():
        relevances[int(q_id)] = []
        for d_id, _ in info.items():
            relevances[int(q_id)].append(int(d_id))

    for i in range(1, total_queries + 1):
        if(not relevances.__contains__(i)):
            relevances[i] = []

    return relevances

def save_words_info2(wembedding, file_name, pdocs, pqueries):
    all_docs = pdocs + pqueries
    model = api.load(wembedding)
    save_word2vect(all_docs, model, file_name)


def main():
    print("load dataset")
    docs_d, pdocs = read_all('../dataset/jsons/CRAN.ALL.json')
    queries_d, pqueries = read_qry('../dataset/jsons/CRAN.QRY.json')
    print(len(pdocs), len(pqueries))
    save_words_info2('glove-wiki-gigaword-50', './word2vect/cran50.bin', pdocs, pqueries)

if __name__ == "__main__":
    main()
