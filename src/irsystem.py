from preproc import *
from gensim import corpora, models, similarities

class IRSystem(object):
    def __init__(self, dataset):
        self.dataset = dataset

    def process_corpus(self, dataset):
        dictionary, pdocs = word2id_dict(dataset)
        corpus_bow = list_docs2bows(dictionary, pdocs)
        return dictionary, corpus_bow

    def create_model(self, mode):
        loaded_corpus = corpora.MmCorpus('vsm_docs.mm') # Recover the corpus
        if mode == 1:
            model = models.TfidfModel(loaded_corpus)
        return model

    def ranking_function(self, dictionary, query, mode = 1):
        model = self.create_model( mode)
        loaded_corpus = corpora.MmCorpus('vsm_docs.mm')
        index = similarities.MatrixSimilarity(loaded_corpus, num_features=len(dictionary))

        query_v = doc2bows(dictionary, query)
        query_w = model(query_v)

        sim = index(query_w)
        ranking = sorted(enumerate(sim), key=itemgetter(1), reverse=True)
        print(len(ranking))
        for doc, score in ranking:
            print ("[ Score = " + "%.3f" % round(score, 3) + "] " + self.dataset[doc]);
        

    def run_system(self, query):
        dictionary, corpus_bow = self.process_corpus(self.dataset)
        self.ranking_function(dictionary, query, 1)
        print("ok")
        
    