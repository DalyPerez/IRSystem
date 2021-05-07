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
        list_texts = re.split("\(.I \d*\n.W\n.N\n\)",open(user_path).read()) # Split text file with the delimiter, erase first delimiter
        return list_texts
       

def main():
    docs = read_dataset('../dataset/cacm/query.text')
    print(docs)
    # print(len(docs))
    # for d in docs:
    #     print(d)


if __name__ == "__main__":
    main()