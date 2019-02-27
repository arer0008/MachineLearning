# Note: you're note allowed to import any other library.
from __future__ import print_function
from gensim.models import word2vec
from os.path import join, exists, split
import os
import numpy as np
import multiprocessing


def find_most_similar(input_word, embedded_vec):
    """
    Input: one word
    Output: list of top 5 similar words only.
    """
    top_similar_words = []
    # TODO: your implementation here

    # end of your implementation
    return top_similar_words


def semantic_math_finder(w_start1, w_end1, w_end2, embedded_model):
    """
    Target: 'w_start1' is related to 'w_end1', as '?X' is related to 'w_end2'.
    E.g.,
    'king' is related to 'queen', as 'husband' is related to 'wife'.
    :param w_start1: word start 1, e.g., 'king'
    :param w_end1: word end 1, e.g., 'queen'
    :param w_end2: word end 2, e.g., 'wife'
    :embedded_model: your pre-train embedded model
    :return: word start 2, e.g., 'husband'
    """
    w_start2 = ""
    #  TODO: your implementation here

    # end of your implementation
    return w_start2


def train_word2vec(sentence_matrix, vocabulary_inv,
                   num_features=300, min_word_count=1, context=10):
    """
    1. Reload word2vec model if exists
    2. Train word2vec model if doens't exist.
    3. Output: returns model and initial weights for embedding layer.
    4. Input params:
        sentence_matrix # int matrix: num_sentences x max_sentence_len
        vocabulary_inv  # dict {int: str}
        num_features    # Word vector dimensionality
        min_word_count  # Minimum word count
        context         # Context window size
    """
    embedding_weights = {}
    embedding_model = None
    # TODO: your implementation here
    model = word2vec("sentences")

    # end of your implementation
    return embedding_model, embedding_weights
