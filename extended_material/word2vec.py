import jax
import jax.numpy as jnp
from jax import grad, jit, vmap, value_and_grad
from jax import value_and_grad
import jax.nn as nn
from jax.random import PRNGKey as Key
from jax.experimental import optimizers
import nltk
import string
import re
import math
import pickle
import random
import numpy as np
import annoy
from numpy import (
    dot, float32 as REAL, double, array, zeros, vstack,
    ndarray, sum as np_sum, prod, argmax, dtype, ascontiguousarray, frombuffer,
)

#==================================
# DATA
#==================================

def skipgram_examples(corpus, vocab_dict, window=2):
    """ Function to identify center and context words in the provided corpus.
        Examples are only generated for words that are in a position in which 
        sufficient context words are available (window*2).
    
    Args:
        corpus (list): containing each document of the corpus represented
        as a list of tokens
        
        vocab_dict (dict): mapping words to their index representation
        
        window (int): window*2 will be the total number of words considered
        as context; (window) words before and (window) words after the
        selected center word
        
    Returns:
        jax array of indexes representing each center word in the corpus
        jax array of jax arrays representing the indexes of context words
    """

    # lists to store the results
    centers = []
    contexts = []
    
    # iterate over al documents in the corpus
    for doc in corpus:

        center = window
        while center < (len(doc)-window):
            # save the current center word
            centers.append(vocab_dict[doc[center]])
            # create a list to store the context of the current center
            context_words = []
            # search for context
            for i in range(0, (window*2)+1):
                if (center-window+i) != center:
                    context_words.append(vocab_dict[doc[center-window+i]])

            # append all the context words identified
            contexts.append(context_words)
            # update center
            center += 1
                   
    return jnp.array(centers), jnp.array(contexts)


def gen_neg_samples(centers_idxs, contexts_idxs, vocab_idxs, num_ns): 
    """ Function to generate negative samples. The number of negative 
        samples produced for each center word will be equal 
        to: neg_samples*window_size*2
    
    Args:
        center_idx (array): containing the index of the center word
        contexts_idx (array): containing the indexes for the context words
        vocab_idxs (array): indices of all the vocabulary tokens
        num_ns (int): number of desired negatives samples PER (CENTER_i, CONTEXT_j) PAIR
        
    Returns
        - A jnp array with the negative samples for each center word
    """
    
    window_size = np.int(contexts_idxs.shape[1]/2)
    neg_idxs = [random.sample(set(vocab_idxs) - set(context) - set([center.item()]), window_size*num_ns*2) for context, center in zip(contexts_idxs, centers_idxs)]

                
    return jnp.array(neg_idxs)

#==================================
# MODEL
#==================================

def init_params(vocab_size, emb_size, seed):
    """ Function to generate random initial parameter matrices
    
    Args:
        vocab_size (int)
        emb_size (int)
        mean (float): of  normal distribution
        std (float): of normal distribution
        seed (int): to initialize NumPy generator
    
    Returns:
        list with two matrices randomly generated with the specified dimensions
    """
    
     
    # initialize the generator
    #generator = np.random.default_rng(seed)
    
    #W = jnp.array(generator.normal(loc=mean, scale=std, size=(vocab_size, emb_size)))
    #C = jnp.array(generator.normal(loc=mean, scale=std, size=(vocab_size, emb_size)))
    
    # GENSIM initialization: https://github.com/RaRe-Technologies/gensim/blob/b3e820bcc708b95cbedab9627d03a6c34af4ea8c/gensim/models/keyedvectors.py#L2011
    prior_vectors = np.zeros((0, 0))
    target_shape = vocab_size, emb_size
    rng = np.random.default_rng(seed=seed)  # use new instance of numpy's recommended generator/algorithm
    
    W = rng.random(target_shape, dtype=REAL)  # [0.0, 1.0)
    W *= 2.0  # [0.0, 2.0)
    W -= 1.0  # [-1.0, 1.0)

    C = rng.random(target_shape, dtype=REAL)  # [0.0, 1.0)
    C *= 2.0  # [0.0, 2.0)
    C -= 1.0  # [-1.0, 1.0)

    return [jnp.array(W), jnp.array(C)]


@jit
def predict_probs(params, center_idx, contexts_idx):
    """ Estimate the probability of the context words given a center word
    
    Args:
        params (list): containing the parameters of the model
        center_idx (int): index of the center word
        contexts_idx (list): containing the indexes of the context words
    
    Returns:
        jax array with one probability for each context word
    """
    
    # unpack the wegihts matrices: Word embeddings and Context embeddings
    W, C = params[0], params[1]
    
    # get the W-embedding of the center word
    W_center = jnp.take(W, center_idx, axis=0)
    
    # get the C-embedding for the context words
    C_contexts = jnp.take(C, contexts_idx, axis=0)
    
    # similarity score: dot product of word embedding of center word and 
    # context embeddings of context words
    similarities = W_center@C_contexts.T
    
    # finally, in order to transform this similarity into a probability we use
    # the sigmoid function
        
    return jax.nn.sigmoid(similarities)

@jit
def loss_per_example(params, center_idx, contexts_idx, ns_idx, noise=0.000001):
    """ calculate the loss for a center word and it's positive and
        negative examples
    
    Args:
        params (list): containing the parameters of the model
        center_idx (int): index of the context word
        contexts_idx (list): containing the indexes of the contexts words
        ns_idx (jax array): containing the indexes of the negative samples
        noise (int): small quantity to avoid passing zero to the logarithm

    Returns:
        loss for a single example
    """
            
    #----------------------------
    # Loss from positive samples
    #----------------------------
    
    # get the scores for the real context
    preds_pos = predict_probs(params, center_idx, contexts_idx)
    
    # loss for the positive (real) context words
    loss_pos = jnp.sum(jnp.log(preds_pos + noise))
    
    #----------------------------
    # Loss from negative samples
    #----------------------------
    
    # get the scores for all the negative samples
    preds_neg = 1 - predict_probs(params, center_idx, ns_idx)
    
    # loss for the negative samples
    loss_neg = jnp.sum(jnp.log(preds_neg + noise))
    
    return -(loss_pos + loss_neg)

# create a vectorized version of the loss using the vmap function from JAX
# the option "in_axes" indicates over which parameters to iterate
batched_loss = jit(vmap(loss_per_example, in_axes=(None, 0, 0, 0, None)))

@jit
def complete_loss(params, all_center_idx, all_contexts_idx, all_ns_idx, noise):
    """ function to calculate the loss for a batch of data by adding the
        individual losses for each example
    
    Args:
        params (list): containing the parameters of the model
        all_center_idx (list): containing all indexes of center words
        all_contexts_idx (list): containing the indexes for the context words
        all_ns_idx (list): containing all negative samples

    Returns:
        average loss for all examples (float)
    """
        
    # get all losses from the examples
    losses = batched_loss(params, all_center_idx, all_contexts_idx, all_ns_idx, noise)
    
    return jnp.sum(losses)/all_center_idx.shape[0]

# use JAX to create a vesion of the loss function that can handle gradients
# the option "argnums" indicates where the parameters of the model are.
# finally use JIT to speed up computations... All JAX magic in one place

grad_loss = jit(value_and_grad(complete_loss, argnums=0))


#==================================
# AUXILIARY
#==================================

def build_indexer(vectors, num_trees=10):
    """ we will use a version of approximate nearest neighbors
        (ANNOY: https://github.com/spotify/annoy) to build an indexer
        of the embeddings matrix
    """
    
    # angular = cosine
    indexer = annoy.AnnoyIndex(vectors.shape[1], 'angular')
    for i, vec in enumerate(vectors):
        # add word embedding to indexer
        indexer.add_item(i, vec)
        
    # build trees for searching
    indexer.build(num_trees)
    
    return indexer

def find_nn(word, word2idx, idx2word, annoy_indexer, n=5):
    """ function to find the nearest neighbors of a given word
    """
    word_index = word2idx[word]
    nearest_indexes =  annoy_indexer.get_nns_by_item(word_index, n+1)
    nearest_words = [idx2word[i] for i in nearest_indexes[1:]]
    
    return nearest_words