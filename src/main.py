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

    docs_dict, pdocs = read_all('../dataset/jsons/CRAN.ALL.json')
    queries_dict, pqueries = read_qry('../dataset/jsons/CRAN.QRY.json')
    relevances = read_rel('../dataset/jsons/CRAN.REL.json', len(pqueries))

    # docs_dict, pdocs = read_all('../dataset/jsons/CISI.ALL.json')
    # queries_dict, pqueries = read_qry('../dataset/jsons/CISI.QRY.json')
    # relevances = read_rel('../dataset/jsons/CISI.REL.json', len(pqueries))

    true, false, relpairs = conforms_pairs(relevances, len(docs_dict))
    print(len(true), len(false), len(relpairs))
    
    r.shuffle(false)

    m = len(false) // 10
    false = false[:m]
    relpairs = true + false
    r.shuffle(relpairs)

    print( len(true), len(false), len(relpairs))
    
    fd = open('./word2vect/cran50.bin')
    w2v_dict = js.load(fd)

    start = time.time()
    TrainSimilarity(docs_dict, queries_dict, relpairs , w2v_dict)
    end = time.time()

    print(end - start)

def testVectModel():
    pdocs, docs_dict = dataset_dict('../dataset/yolanda/corpus/MED.ALL')
    pqueries, queries_dict = dataset_dict('../dataset/yolanda/queries/MED.QRY')
    relevances = read_relevances('../dataset/yolanda/relevance/MED.REL')

    # docs_dict, pdocs = read_all('../dataset/jsons/CRAN.ALL.json')
    # queries_dict, pqueries = read_qry('../dataset/jsons/CRAN.QRY.json')
    # relevances = read_rel('../dataset/jsons/CRAN.REL.json', len(pqueries))

    # docs_dict, pdocs = read_all('../dataset/jsons/CISI.ALL.json')
    # queries_dict, pqueries = read_qry('../dataset/jsons/CISI.QRY.json')
    # relevances = read_rel('../dataset/jsons/CISI.REL.json', len(pqueries))

    system = VectSystem(pdocs, relevances)
    
    mode = 1
    queries_count = 0
    for mode in range(1, 5):
        system.reset()
        sp, sr = 0, 0
        for query_id, _ in queries_dict.items():
            queries_count += 1
            system.run_system(queries_dict[query_id], query_id, mode)

            evaluator = IREvaluator(relevances, system.ranking_querys)
            p, r = evaluator.evaluate_query(query_id)
            sp += p
            sr += r
        print("-> Mode ", mode, "precision average: ", float(sp)/queries_count, "recall average: ", float(sr)/queries_count)


def main():
    testLSTMModel()
    # testVectModel()

   
    

if __name__ == "__main__":
    main()