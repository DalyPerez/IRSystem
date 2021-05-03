from preproc import *
from irsystem import *
from evaluator import *
import re
import sys
import os
import csv

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

def read_relevances(rel_path):
    lines = re.split("\n",open(rel_path).read())
    lines = [re.split(' 0 | 1', l) for l in lines]  
    lines.remove([''])
    relevances= {}
    for l in lines:
        if not relevances.__contains__(l[0]): 
            relevances[l[0]] = []
        relevances[l[0]].append(l[1])
    return relevances

def main():
 
    query2docs_rel = read_relevances('../dataset/relevance/MED.REL')
    dataset_text_list = read_dataset('../dataset/corpus/MED.ALL') # a list of the loaded documents in dataset
    query_text_list = read_dataset('../dataset/queries/MED.QRY') # a list of the loaded queries
    
    system = IRSystem(dataset_text_list, query2docs_rel)
    
    for i in range(1, 31):
        query_id = str(i)
        system.run_system(query_text_list[0], query_id, 1)

        evaluator = IREvaluator(query2docs_rel, system.ranking_querys)
        evaluator.evaluate_query(query_id)
   
    


main()