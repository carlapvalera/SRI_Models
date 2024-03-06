from ir_measures import ScoredDoc
import numpy as np
from collections import Counter
from Utils import Query
from vectorial_model import FrequencyModel


class ClassVectorSpaceModel(FrequencyModel):
    """ Implementation of the vector space model studied in class """

    def fit(self, corpus):
        super().fit(corpus)

        # Computes the norms of the documents
        self.doc_norm = {doc: 0 for doc in self.docs}
        for term in self.tf:
            for doc in self.tf[term]:
                self.doc_norm[doc] += (self.tf[term][doc] * self.idf[term]) ** 2
        self.doc_norm = {doc: np.sqrt(norm) for doc, norm in self.doc_norm.items()}

    def dscore(self, term: int, doc: int):
        return self.tf[term][doc] * self.idf[term] / self.doc_norm[doc]

    def qscore(self, query):
        query_tf = Counter(query)
        max_tf = max(query_tf.values())
        w_tq = {}
        norm = 0
        for term in query_tf:
            w = (0.4 + (1 - 0.4) * (query_tf[term] / max_tf)) * self.idf[term]
            w_tq[term] = w
            norm += w ** 2
        norm = np.sqrt(norm)
        qscore = lambda t: w_tq[t]
        return self.rocchio(query, qscore)

    def tf_function(self, term: int, doc: int):
        return self._tf_m(term, doc)

    def idf_function(self, term: int):
        return self._idf_t(term)

    def retrieve_query(self, query: Query):
        query_id = query.id
        # replacing query words for terms to not repeat this step in the loop
        query = [self.vocabulary[word] for word in query if word in self.vocabulary]
        if len(query) == 0:
            return []

        return [
            ScoredDoc(query_id, str(doc), score)
            for score, doc in self.daat_kretrieve(query, 15)
        ]
