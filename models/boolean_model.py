import sympy
from utils import Query, Doc
import spacy
from sympy import sympify, to_dnf
from typing import Iterable
from models.vectorial_model import IRModel
import gensim
from gensim.utils import simple_preprocess


class boolean_model(IRModel):
    tokenized_docs = []
    titles = []
    corpus = None

    def __init__(self) -> None:
        super().__init__()

       

    def query_to_dnf(query):
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(query)

        # Tokenizar la consulta y reemplazar los operadores lógicos con equivalentes sympy
        processed_query = ' '.join([token.text if token.text not in ['AND', 'OR',
                                                                     'NOT'] else '&' if token.text == 'AND' else '|' if token.text == 'OR' else '~'
                                    for token in doc])

        # Convertir a expresión sympy y aplicar to_dnf
        query_expr = sympify(processed_query, evaluate=False)

        query_dnf = to_dnf(query_expr, simplify=True)
        print("simplificar")
        print(query_dnf)
        print("''''''''")
        return query_dnf

    def fit(self, corpus: Iterable[Doc]) -> None:
        """Computes the structures for the tf-idf framework given a corpus of documents

        This method is abstract and is intended to be overriden by a
        class implementation of IRModel

        Args:
            corpus (Iterable[Doc]): Corpus of documents
        """
        for doc in corpus:
            self.tokenized_docs.append(doc.words)
            self.titles.append(doc.id)

        # Representacion vectorial

        dictionary = gensim.corpora.Dictionary(self.tokenized_docs)
        vocabulary = list(dictionary.token2id.keys())
        self.corpus = [dictionary.doc2bow(doc) for doc in self.tokenized_docs]

    def append_query(self, query) -> str:
        return "".join(query)

    def retrieve_query(self, query: Query):
        """Retrieve the relevant documents for a given query

        This method is abstract and is intended to be overriden by a
        class implementation of IRModel

        Args:
            query (Query): User query
        """

        # Función para verificar si un documento satisface una componente conjuntiva de la consulta
        def satisfies_conjunctive_expression(doc, conj_expr):
            for term in conj_expr.args:
                if term.is_Not:
                    if term.args[0] in doc:
                        return False
                else:
                    token = term.name
                    if not token in doc:
                        return False
                    # return True
            return True

        # Función para verificar si un documento satisface la forma normal disyuntiva de la consulta
        def satisfies_dnf(doc, dnf):
            for conj_expr in dnf.args:
                if satisfies_conjunctive_expression(doc, conj_expr):
                    return True
            return False

        # Verificar la similitud entre los documentos y la consulta
        matching_documents = [i for i, doc in enumerate(self.tokenized_docs) if satisfies_dnf(doc, self.query_to_dnf(query))]

        return matching_documents

