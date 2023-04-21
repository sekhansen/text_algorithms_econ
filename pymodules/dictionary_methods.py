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
import nltk
from nltk import SnowballStemmer

import preprocessing_class as pc

# lemmatize with standard spaCy lemmatizer (takes a couple of minutes)
def lemmatize_token(token, nlp_standard):
    lemmatized = nlp_standard(token)
    return lemmatized[0].lemma_


def dict_example(data_path, items, replacing_dict):
    
    # read MPC minutes
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
    
    # initialize the class with the text data and some parameters
    prep = pc.RawDocs(data.minutes)

    # replace some specific phrases of interest
    prep.phrase_replace(replace_dict=replacing_dict, 
                        sort_dict=True,
                        case_sensitive_replacing=False)
    
    # lower-case text and expand contractions
    prep.basic_cleaning(lower_case=True,
                        contraction_split=True)

    # split the documents into tokens
    prep.tokenize_text(tokenization_pattern=pattern)

    # clean tokens (remove non-ascii characters, remove short tokens, remove punctuation and numbers)
    punctuation = string.punctuation.replace("-", "")
    prep.token_clean(length=2, 
                    punctuation=punctuation, 
                    numbers=True)

    # remove stopwords
    stopwords_type = 'long'
    if items == "tokens":
        prep.stopword_remove(items='tokens', stopwords=stopwords_type)
    elif items == "lemmas":
        prep.lemmatize()
        prep.stopword_remove("lemmas", stopwords=stopwords_type)
    elif items == "stems":
        prep.stem()
        prep.stopword_remove("stems", stopwords=stopwords_type)

    prep.token_clean(length=2, punctuation=punctuation, numbers=True)

    # stem and lemmatize tokens
    if items == "stems":
        prep.stem()
    elif items == "lemmas":
        prep.lemmatize()

    # create document-term matrix
    prep.dt_matrix_create(items=items, score_type='df', min_df=4)

    return data, prep


def pos_neg_counts(prep_dict, items, pos_terms, neg_terms, nlp_standard):
    
    if items == "lemmas":
        # lemmatize the dictionary terms
        pos_terms = [lemmatize_token(w, nlp_standard) for w in pos_terms]
        neg_terms = [lemmatize_token(w, nlp_standard) for w in neg_terms]

    elif items == "stems":
        stemmer = SnowballStemmer(language='english')
        pos_terms = [stemmer.stem(t) for t in pos_terms]
        pos_terms = list(set(pos_terms))
        neg_terms = [stemmer.stem(t) for t in neg_terms]
        neg_terms = list(set(neg_terms))

    # build the document-term matrix
    prep_dict.dt_matrix_create(items=items, score_type='df', min_df=10)

    # find the position of the terms
    pos_ixs = [v for k,v in prep_dict.vocabulary[items].items() if k in pos_terms]
    neg_ixs = [v for k,v in prep_dict.vocabulary[items].items() if k in neg_terms]

    # count positive terms
    pos_counts = np.take(prep_dict.df_matrix[items], pos_ixs, axis=1)
    pos_counts = pos_counts.sum(axis=1)

    # count negative terms
    neg_counts = np.take(prep_dict.df_matrix[items], neg_ixs, axis=1)
    neg_counts = neg_counts.sum(axis=1)
    
    return pos_counts, neg_counts