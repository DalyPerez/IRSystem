from preproc import *
from evaluator import *
from load_data import *
import re
import sys
import os
import csv
from IRModel import TrainSimilarity
import time, random

r = random.Random()

r.seed(99)

def main():
    # docs, docs_dict = dataset_dict('../dataset/yolanda/corpus/MED.ALL')
    # queries, queries_dict = dataset_dict('../dataset/yolanda/queries/MED.QRY')
    # relevances = read_relevances('../dataset/yolanda/relevance/MED.REL')

    docs_dict, pdocs = read_all('../dataset/jsons/CISI.ALL.json')
    queries_dict, pqueries = read_qry('../dataset/jsons/CISI.QRY.json')
    relevances = read_rel('../dataset/jsons/CISI.REL.json', len(pqueries))

    true, false, relpairs = conforms_pairs(relevances, len(docs_dict))
    print(len(true), len(false), len(relpairs))
    
    r.shuffle(false)

    m = len(false) // 5
    false = false[:m]
    relpairs = true + false
    r.shuffle(relpairs)

    print( len(true), len(false), len(relpairs))
    
    fd = open('ciri50.bin')
    w2v_dict = js.load(fd)

    start = time.time()
    TrainSimilarity(docs_dict, queries_dict, relpairs , w2v_dict)
    end = time.time()

    print(end - start)

   
    

if __name__ == "__main__":
    main()