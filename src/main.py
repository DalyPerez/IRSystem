from preproc import *
from evaluator import *
from load_data import *
import re
import sys
import os
import csv
from IRModel import TrainSimilarity
import time, random

def main():
    docs, docs_dict = dataset_dict('../dataset/corpus/MED.ALL')
    queries, queries_dict = dataset_dict('../dataset/queries/MED.QRY')
    relevances = read_relevances('../dataset/relevance/MED.REL')

    t, f, relpairs = conforms_pairs(relevances, len(docs_dict))
    
    fd = open('w2vect50.bin')
    w2v_dict = js.load(fd)

    start = time.time()
    
    TrainSimilarity(docs_dict, queries_dict, relpairs , w2v_dict)
    end = time.time()

    print(end - start)

   
    

if __name__ == "__main__":
    main()