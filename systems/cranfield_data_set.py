from systems.irsystem import IRSystem
from utils import Query, Doc
import ir_datasets
import ir_measures


cranfield = ir_datasets.load("cranfield")


class CranfieldIR(IRSystem):
    """Information Retrieval system for Cranfield corpus"""

    def docs_iter(self):
        for doc in cranfield.docs_iter():
            yield Doc(int(doc.doc_id), self.text_processor.process(doc.text))
    
    def get_doc(self, id):
        doc_store =  cranfield.docs_store()
        return doc_store.get(id)

    def retrieve(self, query):
        query = Query(query.query_id, self.text_processor.process(query.text))
        return self.model.retrieve_query(query)

    def eval(self, measures):
        run = []
        for query in cranfield.queries_iter():
             for scored_doc in self.retrieve(query):
                run.append(scored_doc)

        results = ir_measures.calc_aggregate(measures, cranfield.qrels_iter(), run)
        return results
