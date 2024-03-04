#modelo vectorial generalizado para la recuperación de información
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD 
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import wordnet
class MV_Generalizado():
    def __init__(self, documentos, n_componentes=2):
        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = self.vectorizer.fit_transform(documentos)
        self.lsa = TruncatedSVD(n_componentes)
        self.lsa_matrix = self.lsa.fit_transform(self.tfidf_matrix)
        
    def similitud(self, consulta):
        representacion_consulta = self.vectorizer.transform([consulta])
        similitudes = cosine_similarity(representacion_consulta, self.tfidf_matrix)
        print(similitudes)  
        for termino, similitud in zip(self.vectorizer.get_feature_names(), similitudes[0]):
            sinonimos = wordnet.synsets(termino)
            for sinonimo in sinonimos:
                similitud += cosine_similarity(representacion_consulta, self.vectorizer.transform([sinonimo.name()]))
        return similitudes
    
    def similitud_lsa(self, consulta):
        representacion_consulta = self.vectorizer.transform([consulta])
        return cosine_similarity(representacion_consulta, self.lsa_matrix)
    
    def documentos_similares(self, consulta, nombres_documentos):
        similitudes = self.similitud(consulta)
        documentos_ordenados = sorted(zip(nombres_documentos, similitudes[0]), key=lambda x: x[1], reverse=True)
        return documentos_ordenados
    
    def documentos_similares_lsa(self, consulta, nombres_documentos):
        similitudes = self.similitud_lsa(consulta)
        documentos_ordenados = sorted(zip(nombres_documentos, similitudes[0]), key=lambda x: x[1], reverse=True)
        return documentos_ordenados
# Ejemplo de uso
documentos = [
    "El análisis semántico es interesante",
    "El procesamiento de lenguaje natural es emocionante",
    "Los algoritmos de aprendizaje automático son poderosos"
]
nombres_documentos = ["documento1", "documento2", "documento3"]
consulta = "Análisis de datos en Python"
mv = MV_Generalizado(documentos)
print("Documentos similares:")
for doc, sim in mv.documentos_similares(consulta, nombres_documentos):
    print(f"Documento: {doc}, Similitud: {sim:.4f}")
print("\nDocumentos similares (LSA):")
for doc, sim in mv.documentos_similares_lsa(consulta, nombres_documentos):
    print(f"Documento: {doc}, Similitud: {sim:.4f}")
# Documentos similares:
# Documento: documento1, Similitud: 0.0000
# Documento: documento2, Similitud: 0.0000
