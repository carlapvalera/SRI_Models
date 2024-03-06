from systems.irsystem import IRSystem
from utils import Query, Doc
import ir_datasets
import ir_measures


trec_covid = ir_datasets.load("cord19/trec-covid/round1")

class TrecCovidIR(IRSystem):
    """Information Retrieval system for TrecCovid Round1 corpus"""

    def docs_iter(self):
        for doc in trec_covid.docs_iter():
            yield Doc(doc.doc_id, self.text_processor.process(doc.title + doc.abstract))
    
    def get_doc(self, id):
        doc_store =  trec_covid.docs_store()
        return doc_store.get(id)

    def retrieve(self, query):
        query = Query(query.query_id, self.text_processor.process(query.description))
        return self.model.retrieve_query(query)

    def eval(self, measures):
        run = []
        for query in trec_covid.queries_iter():
             for scored_doc in self.retrieve(query):
                run.append(scored_doc)

        results = ir_measures.calc_aggregate(measures, trec_covid.qrels_iter(), run)
        return results
