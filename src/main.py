from preproc import *
from evaluator import *
from load_data import *
import re
import sys
import os
import csv
from VectModel import VectSystem
from LSTMModel import TrainSimilarity
import time, random

r = random.Random()

r.seed(99)

def testLSTMModel():
    # docs, docs_dict = dataset_dict('../dataset/yolanda/corpus/MED.ALL')
    # queries, queries_dict = dataset_dict('../dataset/yolanda/queries/MED.QRY')
    # relevances = read_relevances('../dataset/yolanda/relevance/MED.REL')

    docs_dict, pdocs = read_all('../dataset/jsons/CISI.ALL.json')
    queries_dict, pqueries = read_qry('../dataset/jsons/CISI.QRY.json')
    relevances = read_rel('../dataset/jsons/CISI.REL.json', len(pqueries))

    true, false, relpairs = conforms_pairs(relevances, len(docs_dict))
    print(len(true), len(false), len(relpairs))
    
    r.shuffle(false)

    m = len(false) // 10
    false = false[:m]
    relpairs = true + false
    r.shuffle(relpairs)

    print( len(true), len(false), len(relpairs))
    
    fd = open('./word2vect/w2vect50.bin')
    w2v_dict = js.load(fd)

    start = time.time()
    TrainSimilarity(docs_dict, queries_dict, relpairs , w2v_dict)
    end = time.time()

    print(end - start)

def testVectModel():
    # query2docs_rel = read_relevances('../dataset/relevance/MED.REL')
    # dataset_text_list = read_dataset('../dataset/corpus/MED.ALL') # a list of the loaded documents in dataset
    # query_text_list = read_dataset('../dataset/queries/MED.QRY') # a list of the loaded queries
    
    docs_dict, pdocs = read_all('../dataset/jsons/CISI.ALL.json')
    queries_dict, pqueries = read_qry('../dataset/jsons/CISI.QRY.json')
    relevances = read_rel('../dataset/jsons/CISI.REL.json', len(pqueries))

    system = VectSystem(pdocs, relevances)
    
    for query_id in range(1, len(relevances) + 1):
        system.run_system(queries_dict[query_id], query_id, 1)

        evaluator = IREvaluator(relevances, system.ranking_querys)
        evaluator.evaluate_query(query_id)

def main():
    # testLSTMModel()
    testVectModel()

   
    

if __name__ == "__main__":
    main()