import gensim
import spacy
from sympy import sympify, to_dnf, Not, And, Or
import Load_data

nlp = spacy.load("en_core_web_sm")
load_data = Load_data()


tokenized_docs = load_data.get_tokenized_docs()

dictionary = gensim.corpora.Dictionary(tokenized_docs)
vocabulary = list(dictionary.token2id.keys())
corpus = [dictionary.doc2bow(doc) for doc in tokenized_docs]


print (corpus)