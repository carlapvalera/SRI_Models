import re
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

class TextProcessor:
    def process(self, text, verbose=False):
        raise NotImplementedError()


class SimpleTextProcessor:
    def __init__(self):
        self.punctuation_table = str.maketrans("", "", string.punctuation)
        self.stop_words = set(stopwords.words("english"))
        self.lemmatizer = WordNetLemmatizer()

    def process(self, text, verbose=False):
        # remove extra withespaces
        text = re.sub(" +", " ", text)

        # convert unicode characters to ascii
        #text = unidecode.unidecode(text)

        # convert text to lowercase
        text = text.lower()

        # expand the contractions so this words can be removed
        #text = contractions.fix(text)

        # remove punctuation symbols
        text = text.translate(self.punctuation_table)

        # tokenize the text
        words = word_tokenize(text)

        result = []
        for w in words:
            w = w.strip()  # remove possible trailing spaces
            if w in self.stop_words:  # discard stopwords
                continue
            w = self.lemmatizer.lemmatize(w)  # lemmatize the word
            if w not in self.stop_words:  # discard stopwords
                result.append(w)

        return result
