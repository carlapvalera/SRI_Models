import ir_datasets
import Json
import spacy
import VectorSpace

class Load_data:

    def tokenization_spacy(self,texts):
        return [[token for token in self.nlp(doc)] for doc in texts]
    
    def create_total_words(tokenized_docs:list[list]):
        total_words = set()
        for doc in tokenized_docs:
            for word in doc:
                total_words.add(word)
        return list(total_words)
                
    
    ''' def create_matrix(total_words,tokenized_docs):
        matrix = []
        for doc in tokenized_docs:
            row = []
            for word in total_words:
                if word in doc:
                    row.append(1)
                else:
                    row.append(0)
            matrix.append(row)
        return matrix
    '''    

    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        dataset = ir_datasets.load("cranfield")
        documents = [doc.text for doc in dataset.docs_iter()]
        self.documents = documents
        tokenized_docs = self.tokenization_spacy(documents)
        self.tokenized_docs = tokenized_docs
        self.total_words = self.create_total_words(tokenized_docs)
        self.vectorial_model = VectorSpace.VectorSpace(tokenized_docs,self.total_words)
        #self.dictionary = self.create_dictionary(self.tokenized_docs)
        #self.vocabulary = self.create_vocabulary(self.dictionary
        self.matrix = self.vectorial_model.tf_idf

    def get_documents(self):
        return self.documents
    def get_tokenized_docs(self):
        return self.tokenized_docs
    def get_total_words(self):  
        return self.total_words
    def get_matrix(self):
        return self.matrix
     
