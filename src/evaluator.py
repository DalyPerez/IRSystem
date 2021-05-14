
class IREvaluator(object):
    def __init__(self, relevance_docs, ranking_querys):
        self.relevance_docs = relevance_docs
        self.ranking_querys = ranking_querys

    def relevant_doc_retrieved(self, query_id):
        true_positives = 0
        false_positives = 0

        query_rel = self.relevance_docs[query_id]
        ranking = self.ranking_querys[query_id]
        ranking = [(d, s) for (d, s) in ranking if s > 0.0]
  
        for (d, s) in ranking: 
            if d in query_rel:
                true_positives +=1 
            else:
                false_positives += 1
        return ranking, true_positives, false_positives             

    def get_precision(self, true_positives, recovered):
        return float(true_positives)/float(recovered)

    def get_recall(self, true_positives, relevant_docs ):
        if (relevant_docs == 0):
            return 0
        return float(true_positives)/ float(relevant_docs)

    def evaluate_query(self, query_id):
        rank, true_pos, false_pos = self.relevant_doc_retrieved(query_id)
        q_relevants_docs = len(self.relevance_docs[query_id])

        precision = self.get_precision(true_pos, len(rank))
        recall = self.get_recall(true_pos, q_relevants_docs)

        print('*** Results Query ', query_id, ' ***')
        print('Precision: ', precision, 'Recall: ', recall)

        return precision, recall

    def get_similarity(self, query_id ):
        """
        Get the similarity of real relevants docs recovered
        """
        print('helllooo')
        if not self.ranking_querys.__contains__(query_id):
            print('The query ', query_id, 'not exists in the system')
        else:
            q_rel_docs = self.relevance_docs[query_id]

            q_rank, _, _ = self.relevant_doc_retrieved(query_id)
            q_rank = [d for (d, s) in q_rank]
           

        
            

            

           
        
    