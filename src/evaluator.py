
class IREvaluator(object):
    def __init__(self, relevance_docs, ranking_querys):
        self.relevance_docs = relevance_docs
        self.ranking_querys = ranking_querys

    def relevant_doc_retrieved(self, query_id):
        true_positives = 0
        false_positives = 0

        query_rel = self.relevance_docs[query_id]
        print(query_rel)
        ranking = self.ranking_querys[query_id]
        ranking = [(d, s) for (d, s) in ranking if s > 0.0]
        print(ranking)
        for (d, s) in ranking: 
            if str(d) in query_rel:
                true_positives +=1 
            else:
                false_positives += 1
        return ranking, true_positives, false_positives             


            

            

           
        
    