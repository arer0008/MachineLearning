# Note: you're note allowed to import any other library.
from __future__ import print_function
from gensim.models import Word2Vec
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
    top_similar_words = embedded_vec.most_similar(positive=input_word, topn=5)
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
    w_start2 = embedded_model.most_similar_cosmul(positive=[w_end1, w_end2], negative=[w_start1], topn=1)
    # end of your implementation
    print(w_start2)
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

    # TODO: your implementation here
    embedding_model = Word2Vec.load("word2vec.model")
    embedding_weights = {}
    sentences = []

    for sentence in sentence_matrix:
        list = []
        for word in sentence:
            list.append(vocabulary_inv.get(word))
        sentences.append(list)



    if embedding_model == None:
        embedding_model = Word2Vec(sentences, size=num_features, window=context, min_count=min_word_count, workers=4)
        embedding_model.save("word2vec.model")
        embedding_model.train(sentences, total_examples=len(sentences), epochs=4)


    # end of your implementation
    return embedding_model, embedding_weights
