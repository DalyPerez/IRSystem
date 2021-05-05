from preproc import *
from evaluator import *
from load_data import *
import re
import sys
import os
import csv

def main():
    docs = dataset_dict('../dataset/corpus/MED.ALL')
    querys = dataset_dict('../dataset/queries/MED.QRY')
    relevances = read_relevances('../dataset/relevance/MED.REL')
    pairs = conforms_pairs(relevances, len(docs))

    model = api.load('glove-wiki-gigaword-300')

    data2train(docs, querys, pairs, model)
   
    

if __name__ == "__main__":
    main()