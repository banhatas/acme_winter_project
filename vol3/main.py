import os
import time
import pickle
import warnings
import numpy as np
import pandas as pd

from tqdm import tqdm   

# files to import
import data_cleaner
import analysis

# define constants
DATA_NAMES = ['twitter', 'yelp', 'imdb']
SUBSET_SIZE = 5000

def get_dataset(filename):
    '''Get a dataset given a filename'''
    return pd.read_csv(filename, index_col = 0)

def clean_data_exists(filename):
    '''Check if cleaned data exists.'''
    assert filename[-4:] == ".pkl", f"Clean data will be saved using the .pkl extension; got {filename}"
    return os.path.isfile(filename)

def save_clean_data(dataset, word_dictionary, file_prefix, dir = '', overwrite = False):
    '''
    Given a cleaned dataset, store it using pickle
    
    Params:
    - dataset : pandas.DataFrame
        a dataset with index valued sequences instead of strings
    - word_dictionary : list
        a list of strings whose indices correspond with the given dataset
    - file_prefix : str
        the name of the file not including the file type (will be saved as file_prefix.pkl)
    - dir : str
        directory/path to save the file to
        must end with '/'
    - overwrite : bool
        overwrites the current data if the file name already exists
    '''

    # asserts
    assert file_prefix in DATA_NAMES, f"file_prefix must be one of {DATA_NAMES}; got {file_prefix}"
    if len(dir) > 0:
        assert dir[-1] == '/', f"dir must end with '/'"

    # handle if the file exists already
    file_name = dir + file_prefix + ".pkl"
    if os.path.isfile(file_name) and not overwrite:
        new_file = dir + file_prefix + "_tmp.pkl"
        print(f"ERROR:\n\t{file_name} already exists. In order to not overwrite the data, will now save data to {new_file}")
        file_name = new_file

    # save the data
    to_dump = {}
    to_dump['data'] = dataset
    if word_dictionary is not None:
        to_dump['dictionary'] = word_dictionary

    with open(file_name, 'wb') as f:
        pickle.dump(to_dump, f)

    print(f"Saved cleaned data and corpus to {file_name}.")


def gridSearch(df, n_clusters, vec_size):
    '''Do a grid search on the pipeline.

    Parameters:
    -----------
    df : pandas.DataFrame
        unclean data
    n_clusters : list
        list of n_clusters to try
    vec_size : list
        list of vec sizes to try
    '''

    loop = tqdm(total = len(n_clusters) * len(vec_size), leave=False, position=0)

    # set up array of accuracies
    mode_accs = np.zeros((len(n_clusters), len(vec_size)))
    final_accs = np.zeros((len(n_clusters), len(vec_size)))

    for i, nc in enumerate(n_clusters):
        for j, vs in enumerate(vec_size):

            # vectorize data
            clean_df = data_cleaner.word2VecCleaner(df, vec_size = vs)

            # cluster data
            clustered_df = data_cleaner.cluster(clean_df, n_clusters=nc)

            # get scores
            acc_mode, acc_final = analysis.analysis(clustered_df, True)
            mode_accs[i, j] = acc_mode
            final_accs[i, j] = acc_final

            # timer
            loop.update()

    return mode_accs, final_accs



if __name__ == "__main__":

    """
    Pipeline Flow
    ----------------------------------
    1. Pull in a csv file and clean it.
    2. Store the clean data in order to be able to access later.
    3. Analyze data (run through modeling and prediction script).
    4. Perform additional analysis.
    5. Write Results
    """

    # file names
    fpath = ''
    fnames = [ 'imdb', 'yelp']

    for i, f in enumerate(fnames):

        # check for clean data
        clean_data_path = f + '.pkl'
        no_clean_data = not clean_data_exists(clean_data_path)

        # if it doesn't exists, clean the data. Otherwise continue to analysis
        if no_clean_data:

            # pull the dataset
            data_filepath = fpath + f + "_data.csv"
            df = get_dataset(data_filepath)

            # DEBUG: get a small sample of the twitter dataset
            df = df.sample(SUBSET_SIZE)

            print(f"DATA CLEANING: Cleaning data from {data_filepath}.")
            start = time.time()
            clean_df = data_cleaner.word2VecCleaner(df, data_name=f)
            clustered_df = data_cleaner.cluster(clean_df, data_name=f)
            save_clean_data(clustered_df, None, f)

        else:
            print(f"DATA CLEANING: Using cleaned data from {clean_data_path}")

            # load the data
            with open(clean_data_path, 'rb') as f:
                data = pickle.load(f)

            try:
                clean_df, corpus = data['data'], data['dictionary']
            except:
                clean_df = data['data']

        # do analysis
        analysis.analysis(clean_df)
        print(f"Finished with {clean_data_path}.\n\n")

        # # ====== grid search =======
        # data_filepath = fpath + f + '_data.csv'
        # df = get_dataset(data_filepath)

        # df = df.sample(SUBSET_SIZE)

        # # perform gridsearch
        # print(f"\nStarting grid search for {f} data\n")
        # cluster_list, vec_list = [5,10,50,100,300], [10, 25, 100, 200]
        # mode_accs, final_accs = gridSearch(df, cluster_list, vec_list)

        # # get best performing models
        # best_mode = np.max(mode_accs)
        # best_final = np.max(final_accs)

        # print(f"Best Accuracy from Mode Method:\t{best_mode:.4f}")
        # print(f"Best Accuracy from Final Method:\t{best_final:.4f}")
