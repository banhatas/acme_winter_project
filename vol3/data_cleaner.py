import pandas as pd
from io import StringIO
from html.parser import HTMLParser
import string
import re
import os
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords

# clean up terminal messages
if not os.path.exists(os.path.expanduser("~") + "/nltk_data/"):
    from nltk import download
    download('stopwords')

# TODO: Correcting misspelled words
#   Not feasible. For every single word, we would
#   have to compute the distance between it and 8,320 
#   other words which causes the temporal complexity
#   to be enormous

# Define stopwords and remove them from the list
# Also reduce words to the root word
def remove_stopwords(text, root='stem'):
    stopword_lst = stopwords.words('english')
    if root == 'stem':
        ps = PorterStemmer()
        return [word if (word == 'acmeuser') or (word == 'acmeurl') else ps.stem(word) for word in text if word not in stopword_lst]
    elif root == 'lemmatize':
        wnl = WordNetLemmatizer()
        return [word if (word == 'acmeuser') or (word == 'acmeurl') else wnl.lemmatize(word) for word in text if word not in stopword_lst]

# Splits the message on one or more non-word character
# Returns as a list
def tokenize(text):
    return re.split("\W+", text)

# Removes punctuation and numbers (by character) and returns as a single string
def remove_punctuation(text):
    return ''.join([char for char in text if (char not in string.punctuation) and (not char.isdigit())])

# Replace usernames from a string with acmeuser
def remove_users(text, replacement_text="acmeuser"):
    user_pattern = re.compile(r'@\S+')
    return user_pattern.sub(replacement_text, text)

# Replace URLs from a string with acmeurl
def remove_urls(text, replacement_text="acmeurl"):
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

def data_cleaner(filthy_data, root='stem'):
    # Apply cleaning functions to the data
    clean_data = filthy_data['text'].apply(lambda x: remove_stopwords( # Remove stopwords and shorten to root words
                                                    tokenize(                   # Split message into a list
                                                    remove_punctuation(         # Remove punctuation and numbers
                                                    remove_users(               # Replace users
                                                    remove_urls(                # Replace URLs
                                                    strip_tags(x)               # Remove HTML tags
                                                    ))).lower()), root))

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
