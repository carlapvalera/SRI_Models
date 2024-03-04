import numpy as np
class LatentSemanticAnalysisModel:
    def __init__(self, documents, window_size=5):
        self.window_size = window_size
        self.documents = documents
        self.co_occurrence_matrix = {}  # Co-occurrence matrix
        self.U, self.S, self.V = None, None, None  # SVD matrices

    def calculate_co_occurrence_matrix(self):
        for document in self.documents:
            for i in range(len(document) - self.window_size + 1):
                for j in range(i + 1, i + self.window_size):
                    term1 = document[i]
                    term2 = document[j]
                    self.co_occurrence_matrix[(term1, term2)] = self.co_occurrence_matrix.get((term1, term2), 0) + 1

    def decompose_svd(self):
        # Convert co-occurrence matrix to numpy array
        co_occurrence_matrix_np = np.array([[self.co_occurrence_matrix.get((i, j), 0) for j in self.co_occurrence_matrix] for i in self.co_occurrence_matrix])
        # Perform SVD
        self.U, self.S, self.V = np.linalg.svd(co_occurrence_matrix_np)

    def calculate_document_vectors(self):
        self.document_vectors = {}
        for i in range(len(self.U-1)):
            for j in range(len(self.documents[i]-1)):
                self.document_vectors[self.documents[i][j]] = self.U[i]

    def search(self, query, k=10):
        query_vector = self.get_query_vector(query)
        # Calculate cosine similarity between query vector and document vectors
        similarities = {}
        for document, document_vector in self.document_vectors.items():
            similarities[document] = np.dot(query_vector, document_vector) / (np.linalg.norm(query_vector) * np.linalg.norm(document_vector))
        # Sort and return top k documents
        sorted_similarities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        return sorted_similarities[:k]

    def get_query_vector(self, query):
        query_vector = np.zeros(len(self.U[0]))
        for term in query:
            if term in self.V:
                query_vector += self.V[term]
        return query_vector / np.linalg.norm(query_vector)

# Example usage
documents = [
    "This is a document about cats.",
    "This is another document about dogs.",
    "This document is about both cats and dogs.",
]

lsa_model = LatentSemanticAnalysisModel(documents)
lsa_model.calculate_co_occurrence_matrix()
lsa_model.decompose_svd()
lsa_model.calculate_document_vectors()

query = ["cats", "dogs"]
results = lsa_model.search(query)
print(results)

