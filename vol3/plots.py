import numpy as np
from matplotlib import pyplot as plt
from HMM import load_object, save_object


def plot1():
    data, labels = load_object('twitter_cluster.pkl')
    for l in np.arange(10):
        # TODO: use the .model to reclassify the sentences into some words if possible so we can label each of the classes
        pass
    plt.hist(labels, 10)
    plt.show()