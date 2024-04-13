import re
import os
import sys
import umap
import string
import pickle
import numpy as np
import pandas as pd

from tqdm import tqdm 
from io import StringIO
from hdbscan import HDBSCAN, prediction
from nltk.corpus import stopwords
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from html.parser import HTMLParser
from nltk.stem import PorterStemmer, WordNetLemmatizer

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
    stopword_lst.append('')
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
    filthy_data = filthy_data.reset_index()
    clean_df = pd.DataFrame()
    clean_df['text'] = seq_data
    clean_df['target'] = filthy_data['target']

    return clean_df, corpus


def cluster(vector_df, data_name, n_clusters=10, gauss=True):
    '''Cluster the data using kmeans. In order to maintain the sequence order we will need to fit
    on one set of (n, m) data, then cluster each vector by iterating through each vector in each
    row of the dataset.

    NOTES:
    ======
    - more neighbors seemed better
    - cosine metric
    '''

    # reset index
    vector_df = vector_df.reset_index(drop=False, inplace=False, names='original_index')

    # turn the data into one long set
    arrays = vector_df['text'].to_list() # turns into a list of lists of arrays
    dat = list()
    lens = list()
    for a in arrays:
        for vec in a:
            dat.append(vec)
        lens.append(len(a))
    dat = np.array(dat)

    # training the model 
    if gauss:
        # transform with umap, then fit with kmeans
        reducer = umap.UMAP(metric='cosine', n_neighbors = 200)
        dat = reducer.fit_transform(dat)

        model = GaussianMixture(n_components=n_clusters)
        model.fit(dat)
    else:
        # transform with umap, then fit with kmeans
        reducer = umap.UMAP(metric='cosine', n_neighbors = 200)
        dat = reducer.fit_transform(dat)

        model = KMeans(n_clusters=n_clusters, random_state = 427, n_init='auto')
        model.fit(dat)


    # with open(data_name + "_cluster.pkl", 'wb') as f:
    #     pickle.dump((dat, model.labels_), f)

    # transform the data
    # print("DATA CLEANING: Clustering the data")
    def get_cluster(row):

        seq = row['text']
        idx = row.name

        if len(seq) == 0:
            return np.array([])
        
        dat_idx_start = sum(lens[:idx])
        dat_idx_end = dat_idx_start + lens[idx]
        to_cluster = dat[dat_idx_start:dat_idx_end]

        clustered_seq = model.predict(to_cluster).astype(int)
        return clustered_seq
    
    vector_df['text'] = vector_df.apply(get_cluster, axis=1)

    # get rid of empty sequences
    vector_df = vector_df.loc[vector_df['text'].apply(len) > 0]

    # print("DATA CLEANING: Clustering finished")

    vector_df.set_index('original_index', inplace=True)

    return vector_df

def hdb_cluster(vector_df, min_cluster_size=5):

    # turn the data into one long set
    arrays = vector_df['text'].to_list() # turns into a list of lists of arrays
    dat = list()
    for a in arrays:
        dat += a
    dat = np.array(dat)

    # train the model
    print("DATA CLEANING: Training clustering model")
    model = HDBSCAN(min_cluster_size=5)
    labels = model.fit_predict(dat)

    # reshape the data
    # TODO: agghhhhh

    # transform the data
    n_clusters = len(model.cluster_persistence_)
    print(f"DATA CLEANING: Clustering the data on {n_clusters} clusters")
    loop = tqdm(total=dat.size, position=0, leave=False)
    def get_cluster(seq, labels):

        sn = len(seq)

        if sn == 0:
            return np.array([])

        clustered_seq = labels[:sn]
        labels = labels[sn:]
        return clustered_seq
    
    vector_df['text'] = vector_df['text'].apply(get_cluster)

    # get rid of empty sequences
    vector_df = vector_df.loc[vector_df['text'].apply(len) > 0]

    return vector_df


# make a word vectorization class
def word2VecCleaner(filthy_data, data_name, root='stem',
                    vec_size=500, window=5, min_count=2):
    """Clean the data and then use word2vec to transform into vectors.
    Once vectors, run kmeans clustering.
    """

    # Apply cleaning functions to the data
    filthy_data['cleaned'] = filthy_data['text'].apply(lambda x: remove_stopwords( # Remove stopwords and shorten to root words
                                                    tokenize(                   # Split message into a list
                                                    remove_punctuation(         # Remove punctuation and numbers
                                                    remove_users(               # Replace users
                                                    remove_urls(                # Replace URLs
                                                    strip_tags(x)               # Remove HTML tags
                                                    ))).lower()), root))
    
    with open("original_twitter_data.pkl", 'wb') as f:
        pickle.dump(filthy_data, f)

    # train the word2vec model
    sentences = filthy_data['cleaned'].to_list()
    model = Word2Vec(sentences,
                     vector_size=vec_size,
                     window=window,
                     min_count=min_count)
    
    # convert text to word2vec embeddings
    def txt_embed(text):
        sentence = list()
        for word in text:
            if word in model.wv:
                sentence.append(model.wv[word])
        return sentence
    
    # apply to data
    clean_df = pd.DataFrame()
    clean_df['text'] = filthy_data['cleaned']
    clean_df['text'] = clean_df['text'].apply(txt_embed)
    clean_df['target'] = filthy_data['target']

    return clean_df


# DEBUGING
if __name__ == "__main__":
    
    word2VecCleaner()