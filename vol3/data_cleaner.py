import pandas as pd
import numpy as np
from io import StringIO
from html.parser import HTMLParser
import string
import pickle
import re
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import download
download('stopwords')

# Removes punctuation and numbers (by character) and returns as a single string
def remove_punctuation(text):
    return ''.join([char for char in text if (char not in string.punctuation) and (not char.isdigit())])

# Remove URLs from a string
def remove_urls(text, replacement_text=""):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(replacement_text, text)

# Remove HTML from a string
class MLStripper(HTMLParser):
    def __init__(self):
        super().__init__()
        self.reset()
        self.strict = False
        self.convert_charrefs= True
        self.text = StringIO()
    def handle_data(self, d):
        self.text.write(d)
    def get_data(self):
        return self.text.getvalue()
def strip_tags(html):
    s = MLStripper()
    s.feed(html)
    return s.get_data()

# Splits the message on one or more non-word character
# Returns as a list
def tokenize(text):
    return re.split("\W+", text)
    
# Define stopwords and remove them from the list
# Also reduce words to the root word
def remove_stopwords(text, root='stem'):
    stopword_lst = stopwords.words('english')
    if root == 'stem':
        ps = PorterStemmer()
        return [ps.stem(word) for word in text if word not in stopword_lst]
    elif root == 'lemmatize':
        wnl = WordNetLemmatizer()
        return [wnl.lemmatize(word) for word in text if word not in stopword_lst]


def data_cleaner(filthy_data, root='stem'):
    # Apply cleaning functions to the data
    clean_data = filthy_data['text'].progress_apply(lambda x: remove_stopwords( # Remove stopwords and shorten to root words
                                                    tokenize(                   # Split message into a list
                                                    remove_punctuation(         # Remove punctuation and numbers
                                                    remove_urls(                # Remove URLs
                                                    strip_tags(x)               # Remove HTML tags
                                                    )).lower()), root))

    # Create our corpus of words
    corpus = []
    for message in clean_data:
        corpus = list(set(corpus + message))
    # Turn each message into a sequence of unique indices
    #   that correspond to a given word
    seq_data = [[corpus.index(word) for word in message] for message in clean_data]
    # Format data as a DataFrame with the original target attached.
    clean_df = pd.DataFrame()
    clean_df['text'] = seq_data
    clean_df['target'] = filthy_data['target']
    
    return clean_df, corpus