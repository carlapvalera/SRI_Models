from typing import Iterable
from utils import Doc
from nlp import TextProcessor
from models.vectorial_model import IRModel
from ir_measures import ScoredDoc, Measure


class IRSystem:
    """An abstract Class for an Information Retrieval System"""

    def __init__(self, model: IRModel, text_processor: TextProcessor, fit=True) -> None:
        self.model = model
        self.text_processor = text_processor
        if fit:
            self.model.fit(self.docs_iter())

    def docs_iter(self) -> Iterable[Doc]:
        """Builds an iterable over the document corpus

        Returns:
            Iterable[Doc]: Document corpus
        """

        raise NotImplementedError()

    def retrieve(self, query) -> list[ScoredDoc]:
        """Retrieves the relevants document in the corpus for a given query

        Args:
            query (Any): The query, the type of this query depends on the implementation

        Returns:
            list[ScoredDoc]: Relevant documents
        """

        raise NotImplementedError()

    def register_query(self) -> int:
        """Assign an unique id to the query for tracking purposes

        Returns:
            int: Query id
        """

        raise NotImplementedError()

    def eval(self, measures: list[Measure]):
        """
        Evaluates the system with the given measures

        Args:
            measures (list[Measure]): Measures for evaluation

        Returns:
            dict[Measure, int | float]: Measure results
        """
        
        raise NotImplementedError()