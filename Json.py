import numpy
import json
class Json:
    def __init__(self,matrix,documents,total_words,tokenized_docs):
        self.total_words = total_words
        self.documents =documents
        self.matrix = matrix
        self.tokenized_docs = tokenized_docs
        self.filename = "data.json" #direccion del json creado

    def get_total_words(self):
        return self.total_words
    def get_documents(self):
        return self.documents
    def get_matrix(self):
        return self.matrix
    def get_tokenized_docs(self):
        return self.tokenized_docs
    
    def to_json(self):
        return {
            "total_words": self.total_words,
            "documents": self.documents,
            "tokenized_docs": self.tokenized_docs,
            "matrix": self.matrix}

    def save(self):
        with open(self.filename, 'w') as file:
            file.write(json.dumps(self.to_json()))
    
    def load(filename):
        with open(filename, 'r') as file:
            return json.load(file)


