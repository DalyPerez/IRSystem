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
    docs, docs_dict = dataset_dict('../dataset/yolanda/corpus/MED.ALL')
    queries, queries_dict = dataset_dict('../dataset/yolanda/queries/MED.QRY')
    relevances = read_relevances('../dataset/yolanda/relevance/MED.REL')

    true, false, relpairs = conforms_pairs(relevances, len(docs_dict))

    
    r.shuffle(false)

    m = len(false) // 10
    false = false[:m]
    relpairs = true + false
    r.shuffle(relpairs)

    print( len(true), len(false), len(relpairs))
    
    fd = open('w2vect50.bin')
    w2v_dict = js.load(fd)

    start = time.time()
    TrainSimilarity(docs_dict, queries_dict, relpairs , w2v_dict)
    end = time.time()

    print(end - start)

   
    

if __name__ == "__main__":
    main()