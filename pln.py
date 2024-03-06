import re
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

class TextProcessor:
    """
    Base class for text processing.
    Subclasses should implement the process method.
    """
    def process(self, text, verbose=False):
        """
        Processes the given text.

        :param text: The text to process.
        :param verbose: If True, prints additional information.
        :raises NotImplementedError: This method should be implemented by subclasses.
        """
        raise NotImplementedError()


class SimpleTextProcessor:
    """
    Class for simple text processing.
    Inherits from TextProcessor and implements the process method.
    """
    def __init__(self):
        """
        Initializes a new SimpleTextProcessor object.
        """
        self.punctuation_table = str.maketrans("", "", string.punctuation)
        self.stop_words = set(stopwords.words("english"))
        self.lemmatizer = WordNetLemmatizer()

    def process(self, text, verbose=False):
        """
        Processes the given text by removing extra spaces, converting to lowercase,
        expanding contractions, removing punctuation symbols, tokenizing the text
        and lemmatizing the words.

        :param text: The text to process.
        :param verbose: If True, prints additional information.
        :return: A list of processed words.
        """
        # remove extra withespaces
        text = re.sub(" +", " ", text)

        # convert text to lowercase
        text = text.lower()

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