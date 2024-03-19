import os
import pickle
import warnings
import numpy as np
import pandas as pd

# files to import
# import data_cleaner

# define constants
DATA_NAMES = ['twitter', 'yelp', 'imdb']

def get_dataset(filename):
    '''Get a dataset given a filename'''
    return pd.read_csv(filename)

def clean_data_exists(filename):
    '''Check if cleaned data exists.'''
    assert filename[-4] == ".pkl", f"Clean data will be saved using the .pkl extension; got {filename}"
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

    print('TODO: data_cleaner.py and analysis.py are not yet implemented.')