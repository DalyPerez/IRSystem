from preproc import *
from irsystem import *
import re
import sys
import os
import csv

def load_dataset(user_path = '../dataset/corpus/MED.ALL'):
    path=user_path[:-8]  # Erase the file name and keep the path
    if os.path.exists(path): # the user has provided a file path with a set of texts
       try:
           list_texts = re.split(".I \d*\n.W\n",open(user_path).read())[1:] # Split text file with the delimiter, erase first delimiter
           print(list_texts[0])
           return list_texts
       except IOError:
            print (user_path + " - No such file or directory")
            sys.exit(0)
    else: 
       print('dont exist path')
       return user_path




def main():
    # dataset = [
    #     "Human machine interface for lab abc computer applications",
    #     "A survey of user opinion of computer system response time",
    #     "The EPS user interface management system",
    #     "System and human system engineering testing of EPS",
    #     "Relation of user perceived response time to error measurement",
    #     "The generation of random binary unordered trees",
    #     "The intersection graph of paths in trees",
    #     "Graph minors IV Widths of trees and well quasi ordering",
    #     "Graph minors A survey",
    # ]

    # query = "Human computer interaction"
    # system = IRSystem(dataset)
    # system.run_system(query, 2)
    load_dataset()

main()