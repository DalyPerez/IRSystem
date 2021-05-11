
from preproc import *
from evaluator import *
from load_data import *
import re
import sys
import os
import csv
from IRModel import TrainSimilarity
import time, random

from keras.models import load_model

def main():
    model = load_model("./models/cisi_50_10_15.h5")

    
    docs_dict, pdocs = read_all('../dataset/jsons/CISI.ALL.json')
    queries_dict, pqueries = read_qry('../dataset/jsons/CISI.QRY.json')
    relevances = read_rel('../dataset/jsons/CISI.REL.json', len(pqueries))
    
    #Load the words vectors dict
    fd = open('./word2vect/cisi50.bin')
    w2v_dict = js.load(fd)

    #Processing query
    query_id = 2
    query = pqueries[query_id -1]
    print("Processed query ", query_id, ": ", query)
    vquery = doc2vector(query, w2v_dict)

    print(relevances[query_id])

    #Processing relevants docs
    vdocs = {}
    for d_id in range(1, len(docs_dict) + 1):
        pdoc = docs_dict[query_id]
        vdoc = doc2vector(pdoc, w2v_dict)
        vdocs[d_id] = vdoc
        pair = [np.array([vquery]), np.array([vdoc])]
        solve = model.predict(pair)[0]
        print("doc id ", d_id, " -> ", solve)


        

    

if __name__ == "__main__":
    main()