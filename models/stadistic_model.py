from typing import Counter
from models.vectorial_model import FrequencyModel
from ir_measures import ScoredDoc
from utils import Query
import math


class OkapiBM25Model(FrequencyModel):
    def __init__(self):
        super().__init__()
        self.k1 = 1.2
        self.b = 0.75
        self.k3 = 7

    def pseudo_feedback(self):
        relevants_by_term = {}
        nrelevants_by_term = {}
        VR = set()
        for term in self.idf:
            VRt, VNRt = self.feedback_on_term(term, sets_diff=False, only_lens=False)
            VR.update(VRt)
            relevants_by_term[term] = len(VRt)
            nrelevants_by_term[term] = len(VNRt)

        VR = len(VR)
        for term in self.idf:
            VRt, VNRt = relevants_by_term[term], nrelevants_by_term[term]
            if not VRt or not VNRt:
                return self.idf[term]
            nom = (VRt + 0.5) / (VNRt + 0.5)
            den = (self.df[term] - VRt + 0.5) / (self.corpus_size - self.df[term] - VR + VRt + 0.5)
            self.idf[term] = math.log(nom / den)

    def dscore(self, term: int, doc: int) -> float:
        nom = self.idf[term] * (self.k1 + 1) * self.tf[term][doc]
        den = self.k1 * ((1-self.b) + self.b*(self.doc_len[doc]/self.avg_doc_len)) + self.tf[term][doc]
        return nom / den

    def qscore(self, query):
        query_tf = Counter(query)
        w_tq = {}
        for term, freq in query_tf.items():
            nom = (self.k3 + 1) * freq
            den = self.k3 + freq
            w_tq[term] = nom / den
        return lambda t : w_tq[t]

    def tf_function(self, term: int, doc: int):
        return self._tf_n(term, doc)

    def idf_function(self, term: int):
        return self._idf_p(term, 0.5)

    def retrieve_query(self, query: Query):
        query_id = query.id
        query = [self.vocabulary[word] for word in query if word in self.vocabulary]
        return [
            ScoredDoc(query_id, str(doc), score)
            for score, doc in self.daat_kretrieve(query, 15)
        ]
