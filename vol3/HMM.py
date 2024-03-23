import numpy as np
from hmmlearn import hmm
import pickle, os

def save_object(obj, filename):
    """a function for saving python class objects to a file for later access"""
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)

def load_object(filename):
    if os.path.exists(filename):
        with open(filename, 'rb') as inF:
            return pickle.load(inF)
    else:
        raise ValueError("filename not recognized, check spelling and existence of file")
    
class SentimentModel():
    """A HMM for sentiment analysis of text"""
    def __init__(self, d):
        """Parameters:
            data (ndarray): a matrix where each row is the a sentence of integer labels to be analyzed
            d (dict): a dictionary of integer labels
            label (ndarray): a list of results for the data should be the same length as data's first dimension
        """
        self.data = None
        self.d = d
        self.labels = None
        self.model = None

    def fit(self, data, labels):
        self.model = hmm.CategoricalHMM(n_components=2, n_iter=200, tol=1e-3, n_features=len(self.d), random_state=66)
        for point in data:
            point = np.array(point).reshape(1, -1)
            self.model.fit(point)
        

    def predict(self, data):
        results= []
        for point in data:
            point = np.array(point).reshape(1, -1)
            results.append(self.model.predict(point))
