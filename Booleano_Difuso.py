#modelo booleano difuso
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
class Booleano_Difuso():
    def __init__(self, documentos):
        self.vectorizer = CountVectorizer()
        self.bow_matrix = self.vectorizer.fit_transform(documentos)
        
    def similitud(self, consulta):
        representacion_consulta = self.vectorizer.transform([consulta])
        return cosine_similarity(representacion_consulta, self.bow_matrix)
    
    def documentos_similares(self, consulta, nombres_documentos):
        similitudes = self.similitud(consulta)
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
bd = Booleano_Difuso(documentos)
print("Documentos similares:")
for doc, sim in bd.documentos_similares(consulta, nombres_documentos):
    print(f"Documento: {doc}, Similitud: {sim:.4f}")
# Documentos similares:
# Documento: documento1, Similitud: 0.0000
# Documento: documento2, Similitud: 0.0000
# Documento: documento3, Similitud: 0.0000