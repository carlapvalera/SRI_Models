import pandas as pd
class Json():
    def __init__(self, matrix, path):
       self.matrix = matrix
       self.path = path

    def save(self):
        # Crear una matriz
        #matriz = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

        # Convertir la matriz a un objeto JSON
        self.matrix.to_json(self.path +"\\datamatriz.json")

    def load(self):
        # Abrir el archivo JSON
        # Load the JSON data into a DataFrame
        matriz = pd.read_json(self.path+"\\datamatriz.json")
        #Imprimir la matriz
        print(matriz)
        return matriz