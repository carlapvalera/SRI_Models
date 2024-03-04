import math
class GeneralizedVectorSpaceModel:
    def __init__(self, documents):
        self.documents = documents
        self.term_frequency = {}
        self.inverse_document_frequency = {}
        self.document_vectors = {}
        self.N = documents.__len__()

    def calculate_term_frequency(self):
        for document in self.documents:
            self.term_frequency[document] = {}
            for term in self.documents[document]:
                if term in self.term_frequency[document]:
                    self.term_frequency[document][term] += 1
                else:
                    self.term_frequency[document][term] = 1

    def calculate_inverse_document_frequency(self):
        total_documents = len(self.documents)
        for document in self.documents:
            for term in self.documents[document]:
                if term in self.inverse_document_frequency:
                    self.inverse_document_frequency[term] += 1
                else:
                    self.inverse_document_frequency[term] = 1

        for term in self.inverse_document_frequency:
            self.inverse_document_frequency[term] = math.log(total_documents / self.inverse_document_frequency[term])

    def calculate_document_vectors(self, weighting_scheme="bm25", k1=1.2, b=0.75):
        """
        Calculates document vectors using the specified weighting scheme.

        Args:
            weighting_scheme (str): The weighting scheme to use. Defaults to "bm25".
            k1 (float): BM25 tuning parameter. Defaults to 1.2.
            b (float): BM25 tuning parameter. Defaults to 0.75.

        Raises:
            ValueError: If the provided weighting scheme is not supported.
        """

        for document in self.documents:
            self.document_vectors[document] = {}
            document_length = sum([self.term_frequency[document][term] for term in self.term_frequency[document]])

            for term in self.documents[document]:
                if weighting_scheme == "tf-idf":
                    tf_idf = self.term_frequency[document][term] * self.inverse_document_frequency[term]
                elif weighting_scheme == "bm25":
                    tf = self.term_frequency[document][term]
                    df = self.inverse_document_frequency[term]
                    avg_doc_length = sum([document_length for document in self.documents]) / len(self.documents)
                    bm25 = tf * (math.log((k1 + 1) / (k1 + tf)) * (math.log(self.N / df) + (b * (document_length - avg_doc_length) / avg_doc_length)))
                    self.document_vectors[document][term] = bm25
                else:
                    raise ValueError(f"Unsupported weighting scheme: {weighting_scheme}")

            # Normalize document vectors (optional)
            # magnitude = math.sqrt(sum([value ** 2 for value in self.document_vectors[document].values()]))
            # if magnitude > 0:
            #     for term, weight in self.document_vectors[document].items():
            #         self.document_vectors[document][term] /= magnitude

        # Set total number of documents (N) for BM25
        #self.N = total_documents
        print(self.document_vectors)

    def search(self, query):
        query_vector = {}
        for term in query:
            if term in query_vector:
                query_vector[term] += 1
            else:
                query_vector[term] = 1

        query_length = math.sqrt(sum([query_vector[term] ** 2 for term in query_vector]))

        scores = {}
        for document in self.documents:
            document_vector = self.document_vectors[document]
            dot_product = sum([query_vector[term] * document_vector.get(term, 0) for term in query_vector])
            document_length = math.sqrt(sum([document_vector[term] ** 2 for term in document_vector]))
            cosine_similarity = dot_product / (query_length * document_length+0.000001)
            scores[document] = cosine_similarity

        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_scores

# Ejemplo de uso
documents = {
    "documento1": ["análisis", "semántico", "interesante"],
    "documento2": ["procesamiento", "lenguaje", "natural", "emocionante"],
    "documento3": ["algoritmos", "aprendizaje", "automático", "poderosos"]
}
vsm = GeneralizedVectorSpaceModel(documents)
vsm.calculate_term_frequency()
vsm.calculate_inverse_document_frequency()
vsm.calculate_document_vectors()
query = ["análisis", "datos", "Python"]
print("Documentos similares:")
for doc, sim in vsm.search(query):
    print(f"Documento: {doc}, Similitud: {sim:.4f}")
    