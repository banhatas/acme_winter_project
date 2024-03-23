import os
import time
import pickle
import warnings
import numpy as np
import pandas as pd

# files to import
import data_cleaner

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
    to_dump['dictionary'] = word_dictionary

    with open(file_name, 'wb') as f:
        pickle.dump(to_dump, f)

    print(f"Saved cleaned data and corpus to {file_name}.")



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
    fpath = '~/dat/school/acme/'
    fnames = ['twitter', 'imdb', 'yelp']

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

            print(f"Cleaning data from {data_filepath}.")
            start = time.time()
            clean_df, corpus = data_cleaner.data_cleaner(df)
            # DEBUG: we will overwrite while debugging. Ensure this is false in final runs
            total_time = time.time() - start
            save_clean_data(clean_df, corpus, f)
            print(f"Took {total_time:.2f} seconds.\n")

        else:
            print(f"Using cleaned data from {clean_data_path}")

    # print('TODO: analysis.py is not yet implemented.')