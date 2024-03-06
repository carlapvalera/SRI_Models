

from typing import Iterable



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

