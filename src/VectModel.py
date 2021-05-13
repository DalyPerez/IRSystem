from preproc import word2id_dict, list_docs2bows, doc2bows
from gensim import corpora, models, similarities
from operator import itemgetter
import numpy

class VectSystem(object):
    def __init__(self, pdocs, query_relevances, ranking_top = 50):
        self.pdocs = pdocs
        self.ranking_querys = {} # { query_id -> ranking list[(doc, score)] }
        self.ranking_top = ranking_top

    def reset(self):
        self.ranking_querys = {}

    def process_corpus(self, pdocs):
        dictionary = word2id_dict(pdocs)
        corpus_bow = list_docs2bows(dictionary, pdocs)
        return dictionary, corpus_bow

    def create_model(self, mode):
        loaded_corpus = corpora.MmCorpus('./word2vect/vsm_docs.mm') # Recover the corpus
        if mode == 1:
            model = models.TfidfModel(loaded_corpus)
        elif mode == 2:
            model = models.LsiModel(loaded_corpus) # LSI model
        elif mode == 3:
            model = models.LdaModel(loaded_corpus) # LDA model
        # elif mode == 4:
        #     model = models.LdaMulticore(loaded_corpus) # LDA Multicore model
        # elif mode == 5:
        #     model = models.RpModel(loaded_corpus) # RP model
        elif mode == 4:
            model = models.LogEntropyModel(loaded_corpus) # LogEntropyModel model

        return model

    def ranking_function(self, dictionary, query, query_id, mode = 1):
        model = self.create_model( mode)
        loaded_corpus = corpora.MmCorpus('./word2vect/vsm_docs.mm')
        index = similarities.MatrixSimilarity(loaded_corpus, num_features=len(dictionary))

        query_v = doc2bows(dictionary, query)
        query_w = model[query_v]

        sim = index[query_w]
        ranking = sorted(enumerate(sim), key=itemgetter(1), reverse=True)
        self.ranking_querys[query_id] = ranking[0: self.ranking_top]
        # self.ranking_querys[query_id] = ranking
        
        # for doc, score in ranking:
        #     print ("[ Score = " + "%.3f" % round(score, 3) + "] ", doc);
        

    def run_system(self, query, query_id, mode):
        dictionary, corpus_bow = self.process_corpus(self.pdocs)
        self.ranking_function(dictionary, query, query_id, mode)

def main():
    pass

if __name__ == "__main__":
    main()
       
        
    