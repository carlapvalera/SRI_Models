from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

# Ejemplo de documentos (reemplázalos con tus propios datos)
documents = [
    "El análisis semántico es interesante",
    "El procesamiento de lenguaje natural es emocionante",
    "Los algoritmos de aprendizaje automático son poderosos"
]

# Crear la matriz de términos-documentos (TF-IDF)
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(documents)

# Aplicar LSA (Truncated SVD)
n_components = 2
lsa = TruncatedSVD(n_components)
lsa_matrix = lsa.fit_transform(tfidf_matrix)

# Resultados
print("Componentes latentes:")
print(lsa.components_)
print("\nDocumentos en el espacio latente:")
print(lsa_matrix)


from sklearn.metrics.pairwise import cosine_similarity

# Representación de la consulta (reemplaza con tu consulta)
consulta = "Análisis de datos en Python"

# Calcula la similitud del coseno entre la consulta y los documentos
similitudes = cosine_similarity(representacion_consulta, representacion_documentos)

# Ordena los documentos según la similitud
documentos_ordenados = sorted(zip(nombres_documentos, similitudes[0]), key=lambda x: x[1], reverse=True)

# Imprime los documentos más similares
for doc, sim in documentos_ordenados:
    print(f"Documento: {doc}, Similitud: {sim:.4f}")
