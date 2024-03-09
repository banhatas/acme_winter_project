import pandas as pd
import numpy as np
from io import StringIO
from html.parser import HTMLParser
import string
import re
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import download
download('stopwords')



data_dir = '/home/tylerc/dat/school/acme/'
twitter_df = pd.read_csv(data_dir + "twitter_data.csv", index_col=0)
imdb_df = pd.read_csv(data_dir + "imdb_data.csv", index_col=0)
yelp_df = pd.read_csv(data_dir + "yelp_data.csv", index_col=0)



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



# Apply cleaning functions to the data
twitter_subset = twitter_df.sample(10000)
clean_data = twitter_subset['text'].progress_apply(lambda x: remove_stopwords(   # Remove stopwords and shorten to root words
                                                tokenize(           # Split message into a list
                                                remove_punctuation( # Remove punctuation and numbers
                                                remove_urls(        # Remove URLs
                                                strip_tags(x)       # Remove HTML tags
                                                )).lower())))

# Turn each mesage into a sequence of unique indices
#   that correspond to a given word
words = []
for message in clean_data:
    words = list(set(words + message))

seq_data = [[words.index(word) for word in message] for message in clean_data]