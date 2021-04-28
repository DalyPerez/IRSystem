from preproc import *
from gensim import corpora, models, similarities
from operator import itemgetter

class IRSystem(object):
    def __init__(self, dataset, query_relevances, ranking_top = 50):
        self.dataset = dataset
        self.ranking_querys = {} # { query_id -> ranking list[(doc, score)] }
        self.ranking_top = ranking_top

    def process_corpus(self, dataset):
        dictionary, pdocs = word2id_dict(dataset)
        corpus_bow = list_docs2bows(dictionary, pdocs)
        return dictionary, corpus_bow

    def create_model(self, mode):
        loaded_corpus = corpora.MmCorpus('vsm_docs.mm') # Recover the corpus
        if mode == 1:
            model = models.TfidfModel(loaded_corpus)
        elif mode == 2:
            model = models.LsiModel(loaded_corpus)
        elif mode == 3:
            model = models.LdaModel(loaded_corpus) # LDA model
        elif mode == 4:
            model = models.LdaMulticore(loaded_corpus) # LDA Multicore model
        elif mode == 5:
            model = models.LsiModel(loaded_corpus) # LSI model
        elif mode == 6:
            model = models.RpModel(loaded_corpus) # RP model
        elif mode == 7:
            model = models.LogEntropyModel(loaded_corpus) # LogEntropyModel model

        return model

    def ranking_function(self, dictionary, query, query_id, mode = 1):
        model = self.create_model( mode)
        loaded_corpus = corpora.MmCorpus('vsm_docs.mm')
        index = similarities.MatrixSimilarity(loaded_corpus, num_features=len(dictionary))

        query_v = doc2bows(dictionary, query)
        query_w = model[query_v]

        sim = index[query_w]
        ranking = sorted(enumerate(sim), key=itemgetter(1), reverse=True)
        self.ranking_querys[query_id] = ranking[0: self.ranking_top]
        
        # for doc, score in ranking:
        #     print ("[ Score = " + "%.3f" % round(score, 3) + "] ", doc);
        

    def run_system(self, query, query_id, mode):
        dictionary, corpus_bow = self.process_corpus(self.dataset)
        self.ranking_function(dictionary, query, query_id, mode)
       
        
    