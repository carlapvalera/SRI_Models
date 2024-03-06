from systems.irsystem import IRSystem
from utils import Query, Doc
import ir_datasets
import ir_measures


vaswani = ir_datasets.load("vaswani")


class VaswanidIR(IRSystem):
    """Information Retrieval system for Vaswanid corpus"""

    def docs_iter(self):
        for doc in vaswani.docs_iter():
            yield Doc(int(doc.doc_id), self.text_processor.process(doc.text))

    def get_doc(self, id):
        doc_store =  vaswani.docs_store()
        return doc_store.get(id)

    def retrieve(self, query):
        query = Query(query.query_id, self.text_processor.process(query.text))
        return self.model.retrieve_query(query)

    def eval(self, measures):
        run = []
        for query in vaswani.queries_iter():
            for scored_doc in self.retrieve(query):
                run.append(scored_doc)

        results = ir_measures.calc_aggregate(measures, vaswani.qrels_iter(), run)
        return results
