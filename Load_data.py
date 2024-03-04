# Import libraries
import csv
import os
import re, os
import nltk
from gensim import corpora
import pandas as pd
from gensim.models import TfidfModel
from gensim import similarities
from scipy.cluster import hierarchy
import matplotlib.pyplot as plt

class Process():
    def __init__(self, path):
        self.path = path
        self.txts = []
        self.titles = []
        self.dictionary = {}
        self.allword = set()
     


    def title_text(self):
        global txts 
        global titles 
        count =0
        with open(self.path + "\\dataset.csv","r") as arch:
            lector_csv = csv.reader(arch, delimiter=",")

            for fila in lector_csv:
                count+=1
                self.titles.append(fila[1])
                self.txts.append(fila[5])
                if count ==15:
                    break



    def remove_noise(self):
        texts = [
                [word.lower() for word in doc if word.isalpha()] for doc in self.txts]
        self.txts = texts

    #remove stopwords
    def remove_stopwords(self):
        # Define a list of stop words
        stoplist = set('for a of the and to in to be which some is at that we i who whom show via may my our might as well'.split())
        # Remove tokens which are part of the list of stop words
        self.txts = [[word for word in txt if word not in stoplist] for txt in self.txts]
        

    #tokenize
    def tokenization(self):
        return[t.split(' ') for t in self.txts]

    #morphological_reduction
    def morphological_reduction(tokenized_docs, use_lemmatization=True):
        stemmer = nltk.stem.PorterStemmer()
        return [
            [token.lemma_ if use_lemmatization else stemmer.stem(token.text) for token in doc]
            for doc in tokenized_docs
        ]

    def normalize_text(self,texts):
        # Convert the text to lower case 
        txts_lower_case = [t.lower() for t in texts ]
        self.txts = txts_lower_case
        # Transform the text into tokens 
        txts = self.tokenization()

        #Remove noise
        txts = self.remove_noise()
        # remove stopwords
        txts = self.remove_stopwords()
        #morphological_reduction
        #txts = morphological_reduction(txts)
        # Print the first 20 tokens for the "On the Origin of Species" book
        return self.txts

    #print(normalize_text(txts))

    def all_word(self):
        for doc in self.txts:
            for word in doc:
                self.allword.add(word)



   
   
        
