import pandas as pd
import numpy as np
import pickle

from hmmlearn import hmm

def flatten_list(X):
    # make the data a 1d array
    sequences = []
    seq_lens = []
    for seq in X:
        sequences.extend(seq)
        seq_lens.append(len(seq))

    # reshape
    sequences = np.reshape(sequences, (-1, 1))

    return sequences, seq_lens


def analysis(clean_df, grid_search = False, save=False):
    """
    Initiates a Hidden Markov Model and fits it to the given data.

    Params:
    - clean_df : pandas.DataFrame
        the dataframe of data with 'text' as our sequence of indices, and 'target' as our target
    """

    # DEBUG
    # SUBSET_SIZE = 

    # separate the data
    X, y = clean_df['text'].tolist(), np.array(clean_df['target'])
    states = np.unique(y)
    n_states = len(states)

    # DEBUG: break into subsets
    # indices = np.random.randint(0, len(X), size=SUBSET_SIZE)
    # y = y[indices]
    # X_ = []
    # for i in indices:
    #     X_.append(X[i])
    # X = X_

    # make the data a 1d array
    sequences, seq_lens = flatten_list(X)

    # init the model and fit
    model = hmm.CategoricalHMM(n_components = n_states, random_state=66)
    model.fit(sequences, seq_lens)

    if not grid_search:
        print(f"Model AIC: {np.exp(model.score(sequences, seq_lens)):.4f}")
        print(f"Model BIC: {model.bic(sequences, seq_lens)}")

    # get a list of predicted sequences
    predicted_seqs = []
    for x in X:
        s = np.reshape(x, (-1,1))
        predicted_seqs.append(model.predict(s))

    clean_df['hmm_data'] = predicted_seqs
    with open("twitter_model_data.pkl", 'wb') as f:
        pickle.dump(clean_df, f)
        print(f'Saved model as {f}')


    # DEBUG: see lenght of most common state
    com_state = 0

    # get most predicted state and get score
    final_sentiment = []
    for pred_seq in predicted_seqs:
        uq, counts = np.unique(pred_seq, return_counts=True)
        possible_states = uq[counts == np.max(counts)]

        # DEBUG
        if len(possible_states) > 1:
            com_state += 1
        
        state_idx = np.random.choice(possible_states)
        final_sentiment.append(states[state_idx])
    final_sentiment = np.array(final_sentiment)

    acc_mode = np.mean(final_sentiment == y) * 100
    rdm_guess_percent = com_state / len(X) * 100.
    if not grid_search:
        print(f"Accuracy from Sequence Mode: {acc_mode:.2f}%")
        print(f"Random Guesses: {rdm_guess_percent:.2f}%")

    # predict the last state
    last_state = []
    for pred_seq in predicted_seqs:
        last_state.append(states[pred_seq[-1]])
    last_state = np.array(last_state)


    acc_final = np.mean(last_state == y) * 100.
    if not grid_search:
        print(f"Accuracy from Final State: {acc_final:.2f}%")

    if grid_search: 
        return acc_mode, acc_final

    
"""
============================= Tyler Notes ======================================
TODO: break up the large dataset into n samples and see for what n do we achieve the highest accuracy?

"""