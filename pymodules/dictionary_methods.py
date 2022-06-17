import sys
import pandas as pd
import numpy as np
from scipy import spatial
import time
import nltk
import string
import re
import math
import pickle
import random

import preprocessing_class as pc

def dict_example(data_path):
    
    data = pd.read_csv(data_path, delimiter='\t', header=0, names=['date', 'minutes'])

    # turn date into date format and create quarter column
    data['date'] = pd.to_datetime(data.date, format='%Y%m')
    data['year'] = data.date.dt.year.astype(int)
    data['quarter'] = data.date.dt.quarter.astype(int)
    data['date'] = data['date'].dt.strftime('%Y%m')

    pattern = r'''
              (?x)                # set flag to allow verbose regexps (to separate logical sections of pattern and add comments)
              \w+(?:-\w+)*        # word characters with internal hyphens
              | [][.,;"'?():-_`]  # preserve punctuation as separate tokens
              '''
    
    # initialize object
    prep = pc.RawDocs(data.minutes, stopwords='long', contraction_split=True, tokenization_pattern=pattern)
    # replace expressions
    replacing_dict = {'financial intermediation':'financial-intermediation', 'interest rate':'interest-rate'}
    prep.phrase_replace(replace_dict=replacing_dict)
    # lower-case text, expand contractions and initialize stopwords list
    prep.basic_cleaning()
    # split the documents into tokens
    prep.tokenize_text()
    punctuation = string.punctuation.replace("-", "")
    prep.token_clean(length=2, punctuation=punctuation, numbers=True)
    prep.stopword_remove('tokens')
    prep.stem()
    prep.lemmatize()
    prep.dt_matrix_create(items='lemmas', score_type='df', min_df=4)
    
    return data, prep


def pos_neg_counts(prep_dict, pos_words, neg_words):
    
    pattern = r'''
              (?x)                # set flag to allow verbose regexps (to separate logical sections of pattern and add comments)
              \w+(?:-\w+)*        # word characters with internal hyphens
              | [][.,;"'?():-_`]  # preserve punctuation as separate tokens
              '''
    
    punctuation = string.punctuation.replace("-", "")
    
    pos_clean = pc.RawDocs(pos_words, stopwords='long', contraction_split=True, tokenization_pattern=pattern)
    pos_clean.basic_cleaning()
    pos_clean.tokenize_text()
    pos_clean.token_clean(length=2,punctuation=punctuation, numbers=True)
    pos_clean.lemmatize() 
    pos_clean.stem() 

    neg_clean = pc.RawDocs(neg_words, stopwords='long', contraction_split=True, tokenization_pattern=pattern)
    neg_clean.basic_cleaning()
    neg_clean.tokenize_text()
    neg_clean.token_clean(length=2,punctuation=punctuation, numbers=True)
    neg_clean.lemmatize() 
    neg_clean.stem()

    #pos_lemmas = set([el[0] for el in pos_clean.lemmas])
    #neg_lemmas = set([el[0] for el in neg_clean.lemmas])
    
    pos_lemmas = set([el[0] for el in pos_clean.lemmas if el])
    neg_lemmas = set([el[0] for el in neg_clean.lemmas if el])
    pos_lemmas0 = pos_lemmas
    neg_lemmas0 = neg_lemmas
    pos_lemmas = [l for l in pos_lemmas0 if l not in neg_lemmas0]
    neg_lemmas = [l for l in neg_lemmas0 if l not in pos_lemmas0]

    prep_dict.get_term_ranking(items='lemmas', score_type='df')
    prep_dict.dt_matrix_create(items='lemmas', score_type='df')

    pos_ixs = [v for k,v in prep_dict.vocabulary['lemmas'].items() if k in pos_lemmas]
    neg_ixs = [v for k,v in prep_dict.vocabulary['lemmas'].items() if k in neg_lemmas]

    pos_counts = np.take(prep_dict.df_matrix['lemmas'], pos_ixs, axis=1)
    pos_counts = pos_counts.sum(axis=1)

    neg_counts = np.take(prep_dict.df_matrix['lemmas'], neg_ixs, axis=1)
    neg_counts = neg_counts.sum(axis=1)
    
    return pos_counts, neg_counts