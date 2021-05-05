import re
import sys
import os
import csv
from gensim.models import Word2Vec
import numpy as np
import gensim.downloader as api

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
    id2doc = {}
    count = 0
    for d in docs:
        id2doc[count] = d
        count = count + 1
    return docs, id2doc

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
        for d in range(1, total_docs + 1):
            if d in query_relevances[i]:
                true_pairs.append((i, d, 1)) # (query, doc, rel)
            else:
                false_pairs.append((i, d, 0))
    return true_pairs, false_pairs, true_pairs + false_pairs


def main():
    # d = dataset_dict('../dataset/corpus/MED.ALL') 
    # query2docs_rel = read_relevances('../dataset/relevance/MED.REL')
    # t, f, l = conforms_pairs(query2docs_rel, 1033)
    # print(len(t), len(f), len(l))
   
    model = api.load('glove-wiki-gigaword-300')
    wv = model.get_vector('house')
    print('--->', wv)
    
if __name__ == "__main__":
    main()