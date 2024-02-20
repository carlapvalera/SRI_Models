import gensim
import spacy
from sympy import sympify, to_dnf, Not, And, Or
import math
import numpy as np


class VectorSpace():

    def TF(tokenized_docs:list[list]):
        tf = []
        for docs in tokenized_docs:
            tf_dict = {}
            for word in docs:
                if word in tf_dict:
                    tf_dict[word] += 1
                else:
                    tf_dict[word] = 1
            for words, value in tf_dict.items():
                value = value/len(tokenized_docs[docs])
            
            tf.append(tf_dict)
            return tf
        
    def IDF(tokenized_docs:list[list], total_words, total_docs:int):
        idf = {}
        for word in total_words:
            exist = 0
            for docs in tokenized_docs:
                if word in docs:
                    exist+=1
            
            idf[word] = exist
        return idf
    
    def IDF_total ( idf:dict, total_docs:int):
        for word, value in idf.items():
            idf[word] = math.log((1+total_docs)/(1+value))+1
        return idf

    def create_matrix(total_words,tokenized_docs,tf_list_dict:list[dict],idf_dict:dict):
        matrix = []
        for doc in tokenized_docs:
            row = []
            for word in total_words:
                if word in doc:
                    row.append(tf_list_dict[doc][word]*idf_dict[word])
                else:
                    row.append(0)
            matrix.append(row)
        return matrix


    def __init__(self, tokenized_docs,total_words):
        self.M = len(tokenized_docs)  # number of files in dataset
        self.total_words = total_words
        self.tf_list_dict = self.TF(tokenized_docs)  # returns term frequency
        self.idf_parcial = self.IDF(tokenized_docs,M)
        self.idf_dict = self.IDF_total(self.idf_parcial)  # returns idf scores
        self.tf_idf = self.create_matrix(total_words,tokenized_docs,self.tf_list_dict,self.idf_dict)  # returns tf-idf scores


    def Query_TF_IDF(self,tokenized_query):
        tf_query=set()
        tf_idf = []
        for word in tokenized_query:
            if word in tokenized_query:
                tf_query[word] +=1
            else :
                tf_query[word] = 1
        
        for word in self.total_words:
            if word in tokenized_query:
                tf_idf.append(tf_query*(math.log((1+self.M)/(self.idf_parcial[word]+2))+1))
        return tf_idf



    