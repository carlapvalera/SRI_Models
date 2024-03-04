#modelo booleano

import math
class Booleano_difuso():
  
    def calcular_peso(termino, documento):
        """Calcula el peso de un término en un documento."""
        tf = documento.count(termino)
        idf = math.log(len(documentos) / len([documento for documento in documentos if termino in documento]))
        return tf * idf

    def calcular_similitud(self,documento, consulta):
        """Calcula la similitud entre un documento y una consulta."""
        vector_documento = [self.calcular_peso(termino, documento) for termino in consulta]
        vector_consulta = [self.calcular_peso(termino, consulta) for termino in consulta]
        similitud = sum(vector_documento[i] * vector_consulta[i] for i in range(len(vector_documento)))
        return similitud

    def recuperar_documentos(self,consulta, documentos):
        """Recupera los documentos relevantes para una consulta."""
        similitudes = [(documento, self.calcular_similitud(documento, consulta)) for documento in documentos]
        similitudes.sort(key=lambda x: x[1], reverse=True)
        return [documento for documento, similitud in similitudes]

# Ejemplo de uso
bool= Booleano_difuso()
documentos = ["Este es el primer documento.", "Este es el segundo documento.", "Este es el tercer documento."]
consulta = ["primer", "segundo"]

documentos_relevantes = bool.recuperar_documentos(consulta, documentos)

for documento in documentos_relevantes:
  print(documento)
