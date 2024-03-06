
from typing import Iterable
from Utils import Doc, Query
from ir_measures import ScoredDoc

import pickle
class IRModel:
    def __init__(self) -> None:
        self.feedback = {}

    def save(self, file):
        with open(file, 'wb') as f:
            pickle.dump(self, f)

    def fit(self, corpus: Iterable[Doc]) -> None:
        """Computes the structures for the tf-idf framework given a corpus of documents

        This method is abstract and is intended to be overriden by a
        class implementation of IRModel

        Args:
            corpus (Iterable[Doc]): Corpus of documents
        """

        raise NotImplementedError()

    def retrieve_query(self, query: Query):
        """Retrieve the relevant documents for a given query

        This method is abstract and is intended to be overriden by a
        class implementation of IRModel

        Args:
            query (Query): User query
        """

        raise NotImplementedError()

    def retrieve_feedback(self, query_id: int, wquery: dict[int, float]):
        """Retrieve the relevant documents for a given computed value of the weights of a query

        This method is abstract and is intended to be overriden by a
        class implementation of IRModel

        Args:
            query_id (int): Id of the original query
            wquery (dict[int, float]): Weights of the query improved
        """

        raise NotImplementedError()

    def export_feedback(self, file):
        with open(file, 'wb') as f:
            pickle.dump(self.feedback, f)

    def import_feedback(self, file):
        with open(file, 'rb') as f:
            self.feedback = pickle.load(f)


class FrequencyModel(IRModel):
    """An abstract representation of a Frequency based
    Information Retrieval System.

    Provides methods for easy implementation of frequency models.
    """

    def __init__(self):
        super().__init__()

        # Maps word to an unique int for more efficient computations this unique int
        # will be the representation of a term
        self.vocabulary: dict[str, int] = {}

        # An inverted index of term frequency with the structure
        # { term : { doc : freq } }
        self.tf: dict[int, dict[int, float]] = {}

        # Associates each term with its idf value
        self.idf: dict[int, float] = {}

        # Associates each term with its document frequency, this field is used
        # just for explainability in the formulas
        self.df: dict[int, int] = {}

        # Associates each document with the total number of terms in the doc,
        # this measure is used for normalization purposes
        self.doc_len: dict[int, int] = {}

        # Associantes each document with the largest number of ocurrences of
        # a term in the document, this measure is used for normalization purposes
        self.doc_max_tf: dict[int, int] = {}

        # Associates each document with the average frequency of a term in it
        self.doc_avg_tf: dict[int, float] = {}

        # Total number of documents in the corpus
        self.corpus_size: int = 0

        # Average number of terms in a document in the corpus
        self.avg_doc_len: float = 0

        # contains the ids of the documents present in the corpus
        self.docs: list[int] = []



    #IRSystem Interface



    def fit(self, corpus: Iterable[Doc]) -> None:
        for doc in corpus:

            self.docs.append(doc.id)

            self.doc_len[doc.id] = len(doc)
            self.avg_doc_len += len(doc)
            self.doc_max_tf[doc.id] = 0
            self.doc_avg_tf[doc.id] = 0

            # Construct a dictionary { word : frequency } for the document
            doc_tf = Counter(doc.words)

            for word, freq in doc_tf.items():

                if word not in self.vocabulary:
                    # Register the word as a term
                    term = self.vocabulary[word] = len(self.vocabulary)
                    self.tf[term] = {}
                else:
                    term = self.vocabulary[word]

                self.tf[term][doc.id] = freq
                self.doc_max_tf[doc.id] = max(self.doc_max_tf[doc.id], freq)
                self.doc_avg_tf[doc.id] += freq

            if len(doc_tf) != 0:
                self.doc_avg_tf[doc.id] /= len(doc_tf)

        self.corpus_size = len(self.docs)
        self.avg_doc_len /= self.corpus_size

        for term in self.tf:
            self.df[term] = len(self.tf[term])
            self.idf[term] = self.idf_function(term)

            self.tf[term] = {doc: self.tf_function(term, doc) for doc in self.tf[term]}

        self.feedback = {term: {'rel': {}, 'nrel': {}} for term in self.tf}

    def dscore(self, term: int, doc: int) -> float:
        """Compute the weight of a term in a document [w(t,d)]

        This method is abstract and is intended to be overriden by a
        class implementation of Frequency Model

        Args:
            term (int): Term id
            doc (int): Doc id

        Returns:
            float: w(t,d) value
        """

        raise NotImplementedError()

    def qscore(self, query):
        raise NotImplementedError()

    def normalize(self, score: float, doc: int):
        raise NotImplementedError()

    def retrieve_query(self, query: Query):
        raise NotImplementedError()

    def retrieve_feedback(self, query_id: int, rdocs: list[str], alldocs: list[str]):
        raise NotImplementedError()

    def feedback_on_doc(self, doc, rel='rel'):
        for term, docs in self.tf.items():
            if doc in docs:
                try:
                    self.feedback[term][rel][doc] += 1
                except KeyError:
                    self.feedback[term][rel][doc] = 1

    def feedback_on_term(self, term, sets_diff=True, only_lens=True):
        relevant_docs = set()
        for term in self.feedback:
            for doc in self.feedback[term]['rel']:
                rel_score = self.feedback[term]['rel'][doc]
                try:
                    nrel_score = self.feedback[term]['nrel'][doc]
                except:
                    nrel_score = 0
                rel_score -= nrel_score
                if rel_score > 0:
                    relevant_docs.add(doc)

        if only_lens and sets_diff:
            return len(relevant_docs), self.corpus_size - len(relevant_docs)

        if not only_lens and sets_diff:
            return relevant_docs, set([doc for doc in self.docs if doc not in relevant_docs])

        non_relevant_docs = set([doc for doc in self.feedback[term]['nrel'] if doc not in relevant_docs])

        if not only_lens and not sets_diff:
            return relevant_docs, non_relevant_docs

        if only_lens and not sets_diff:
            return len(relevant_docs), len(non_relevant_docs)

    def rocchio(self, query, qscore, alpha=0.75, beta=0.75, gamma=0.25):
        w_tq = {}
        for term in query:
            previous_w_tq = qscore(term)
            VRt, VNRt = self.feedback_on_term(term, False, False)
            VRt_score = sum(self.dscore(term, doc) for doc in VRt)
            VNRt_score = sum(self.dscore(term, doc) for doc in VNRt)
            w = alpha * previous_w_tq
            if VRt:
                w += VRt_score * beta / len(VRt)
            if VNRt:
                w -= VNRt_score * gamma / len(VNRt)
            if w > 0:
                w_tq[term] = w
            else:
                w_tq[term] = 0

        return lambda t: w_tq[t]





    #Definitions of tf



    def tf_function(self, term: int, doc: int) -> float:
        """
        Computes tf(t,d)

        This method is intended for calling an implementation method of
        tf(t,d) therefore is intended to be overrided in a model implementation class.

        The class provides several of these methods:
        - _tf_n
        - _tf_m
        - _tf_s
        - _tf_l
        - _tf_a
        - _tf_b
        - _tf_L

        The definition of these Methods are extracted from
        Christopher, D.M., Prabhakar, R. and Hinrich, S., 2008. Introduction to information retrieval.

        Args:
            term (int): word id
            doc (int): document id

        Returns:
            float: tf(t,d) value
        """

        raise NotImplementedError()

    def _tf_n(self, term: int, doc: int) -> float:
        """
        Natural definition of tf(t,d)

        ntf(t,d) = tf(t,d)
        """
        return self.tf[term][doc]

    def _tf_m(self, term: int, doc: int) -> float:
        """
        Maximun tf(t,d) normalization

        mtf(t,d) = tf(t,d) / max_t{tf(t,d)}
        """
        return self.tf[term][doc] / self.doc_max_tf[doc]

    def _tf_s(self, term: int, doc: int, a: float = 0.4) -> float:
        """
        Maximun tf(t,d) normalization smooth

        stf(t,d) = 1 + (1 - a) * tf(t,d) / max_t{tf(t,d)}

        Common value for a is 0.4

        Args:
            a (float) [defaults: 0.4]: The smoothing coefficient.
        """
        return a + (1 - a) * self.tf[term][doc] / self.doc_max_tf[doc]

    def _tf_l(self, term: int, doc: int) -> float:
        """
        Logarithm definition of tf(t,d)

        ltf(t,d) = log(1 + tf(t,d))
        """
        return np.log10(1 + self.tf[term][doc])

    def _tf_a(self, term: int, doc: int, a: float = 0.5) -> float:
        """
        Augmented definition of tf(t,d)

        atf(t,d) = a + (a * tf(t,d)) / max_t{tf(t,d)}

        Common values for a are [1, 0.5]

        Args:
            a (float) [defaults: 0.5]: The smoothing coefficient.
        """
        return a + (a * self.tf[term][doc]) / self.doc_max_tf[doc]

    def _tf_b(self, term: int, doc: int) -> float:
        """
        Boolean definition of tf(t,d)

        btf(t,d) = 1 if tf(t,d) > 0
                   0 otherwise
        """
        return 1 if self.tf[term][doc] else 0

    def _tf_L(self, term: int, doc: int) -> float:
        """
        Log ave definition of tf(t,d)

        Ltf(t,d) = (1 + log(tf(t,d))) / (1 + log(avg_d{tf(t,d)}))
        """
        return (1 + np.log10(self.tf[term][doc])) / (1 + np.log10(self.doc_avg_tf[doc]))





    # Definitions of idf



    def idf_function(self, term: int) -> float:
        """
        Computes idf(t)

        This method is intended for calling an implementation method of
        idf(t) therefore is intended to be overrided in a model implementation class.

        The class provides several of these methods:
        - _idf_n
        - _idf_t
        - _idf_p

        The definition of these Methods are extracted from
        Christopher, D.M., Prabhakar, R. and Hinrich, S., 2008. Introduction to information retrieval.

        Args:
            term (int): Word id

        Returns:
            float: idf(t) value
        """

        raise NotImplementedError()

    def _idf_n(self, term: int):
        """
        Represents no idf

        idf(t) = 1
        """
        return 1

    def _idf_t(self, term: int, a: float = 0):
        """
        Vectorial space definition of idf

        idf(t) = log((N + a) / (df(t) + a)) + a

        Chosing a = 0 yields the basic definition of idf(t) = log(N / df(t))
        Another common choice is a = 1 for smoothing the idf.

        Args:
            a (float) [defaults: 0]: The smoothing coefficient.
        """
        return np.log10((self.corpus_size + a) / (self.df[term] + a)) + a

    def _idf_p(self, term: int, a: float = 0):
        """
        Probabilistic definition of idf

        idf(t) = max{0, log((N - df(t) + a) / (df(t) + a))}

        Common values for a are 0 and 0.5

        Args:
            a (float) [defaults: 0]: The smoothing coefficient.
        """
        return max(
            0, np.log10((self.corpus_size + a) / (self.df[term] + a))
        )



    # Retrieval methods



    def daat_kretrieve(self, query, k: int) -> list[ScoredDoc]:
        """Retrieves the k most relevant documents in the corpus for a
        given query using a Document-At-A-Time computation strategy.

        Args:
            wquery (dict[int, float]): Computed value of the weights of the user query
            k (int): Number of documents to retrieve

        Returns:
            list[(float, int)]: k most relevant documents for q in format (score, doc)
        """
        heap = KMaxHeap(k)

        qscore = self.qscore(query)

        for doc in self.docs:
            score = 0
            for term in query:
                if doc in self.tf[term].keys():
                    score += self.dscore(term, doc) * qscore(term)
            heap.push((score, doc))

        return heap.to_list(reverse=True)

    def daat_tretrieve(self, query, t: float) -> list[ScoredDoc]:
        """Retrieves the k most relevant documents in the corpus for a
        given query using a Document-At-A-Time computation strategy.

        Args:
            query (dict[int, float]): Computed value of the weights of the user query
            k (int): Number of documents to retrieve

        Returns:
            list[(float, int)]: k most relevant documents for q in format (score, doc)
        """
        result = []

        qscore = self.qscore(query)

        for doc in self.docs:
            score = 0
            for term in query:
                if doc in self.tf[term]:
                    score += self.dscore(term, doc) * qscore(term)
            if score > t:
                result.append((score, doc))

        return result


    def taat_kretrieve(self, query: Query, k: int) -> list[(float, int)]:
        """Retrieves the k most relevant documents in the corpus for a
        given query using a Term-At-A-Time computation strategy.

        Args:
            query (Query): User query
            k (int): Number of documents to retrieve

        Returns:
            list[(float, int)]: k most relevant documents for q in format (score, doc)
        """
        scores = {}
        heap = KMaxHeap(k)

        # replacing query words for terms to not repeat this step in the loop
        query = [self.vocabulary[word] for word in query if word in self.vocabulary]
        query_tf = Counter(query)
        qscore = self.qscore(query_tf)

        for term in query_tf:

            for doc in self.tf[term]:
                wt = self.dscore(term, doc) * qscore(term)
                try:
                    scores[doc] += wt
                except KeyError:
                    scores[doc] = wt

        for doc, score in scores.items():
            heap.push((score, doc))
        return heap.to_list(reverse=True)
