from preproc import *
from evaluator import *
from load_data import *
import re
import sys
import os
import csv
from operator import itemgetter
import numpy
# from VectModel import VectSystem
# from LSTMModel import TrainSimilarity
import time, random
from keras.models import load_model

r = random.Random()
r.seed(99)

def select_model(m):
    return load_model("./models/lstmmodel_med10-15.h5")
    

    
def select_dataset(n):
    if n == 1:
        pdocs, docs_dict = dataset_dict('../dataset/yolanda/corpus/MED.ALL')
        pqueries, queries_dict = dataset_dict('../dataset/yolanda/queries/MED.QRY')
        relevances = read_relevances('../dataset/yolanda/relevance/MED.REL')
        fd = open('./word2vect/w2vect50.bin')
        w2v_dict = js.load(fd)
    elif n == 2:
        docs_dict, pdocs = read_all('../dataset/jsons/CISI.ALL.json')
        queries_dict, pqueries = read_qry('../dataset/jsons/CISI.QRY.json')
        relevances = read_rel('../dataset/jsons/CISI.REL.json', len(pqueries))
        fd = open('./word2vect/cisi50.bin')
        w2v_dict = js.load(fd)
    else:
        docs_dict, pdocs = read_all('../dataset/jsons/CRAN.ALL.json')
        queries_dict, pqueries = read_qry('../dataset/jsons/CRAN.QRY.json')
        relevances = read_rel('../dataset/jsons/CRAN.REL.json', len(pqueries))
        fd = open('./word2vect/cran50.bin')
        w2v_dict = js.load(fd)
    
    return pdocs, docs_dict, pqueries, queries_dict, relevances, w2v_dict

def execute_query(q_id, vquery, docs_dict, relevances, w2v_dict, model):
    ranking = {}
    r = []
    for d_id, pdoc in  docs_dict.items():
        vdoc = doc2vector(pdoc, w2v_dict)
        pair = [np.array([vquery]), np.array([vdoc])]
        solve = model.predict(pair)[0]
        r.append((d_id, solve[1]))
    r = sorted(enumerate(r), key=itemgetter(1), reverse=True)
    ranking[q_id] = r[0: 20]
    return ranking
        



def main():
    while(1):
        print("\t*** Information Retrieval System***\n")
        print( "\t\tSelect the model option:\n")

        print('\t\t1- Gensim Vectorial Model \n')
        print('\t\t2- LSTM Model\n')
        print('\t\t3- Match Pyramid Model\n')
        print('\t\t0- Exit...\n')

        index = -1

        while 1:
            index = int(input('\t\t > '))

            if index >= 0 and index <= 3:
                break
            else:
                print('\n\t\t Please, select a valid option...\n')
        
        model = index
        if index == 0:
            break

        print( "\n\t\tSelect the dataset to test:\n")

        print('\t\t1- Medline \n')
        print('\t\t2- CISI\n')
        print('\t\t3- Cranfield\n')

        dataset = -1

        while 1:
            index = int(input('\t\t > '))

            if index >= 1 and index <= 3:
                break
            else:
                print('\n\t\t Please, select a valid dataset...\n')
        
        q_id = int(input("\n-> Write a query id: "))
        pdocs, docs_dict, pqueries, queries_dict, relevances, w2v_dict = select_dataset(dataset)
        pquery = queries_dict[q_id]
        print("-- Preprocessed Query: ", pquery)

        
        if model == 1:
            print("Search for result in Vectorial Model...\n\n")

        elif model == 2:
            print("Search for result in LSTM Model...\n\n")
            vquery = doc2vector(pquery, w2v_dict)
            model = select_model(model)
            rank = execute_query(q_id, vquery, docs_dict, relevances, w2v_dict, model)
            evaluator = IREvaluator(relevances, rank)
            p, r = evaluator.evaluate_query(q_id)

        elif  model == 3:
            print("Search for result in Matching Pyramid Model...")

        else:
            break

        
            

        

main()