#!/usr/bin python3
# -*- coding : utf-8 -*
"""

Author: fenia
"""

import collections
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial import distance
from scipy.stats import spearmanr
import numpy as np

NonCntxPair = collections.namedtuple('ncPair', 'word1 word2 score')
CntxPair = collections.namedtuple('cPair', 'word1 word2 cntx1 cntx2 score')


def load_non_contextual(data):
    """
    Load contextual dataset.
    Args:
        data: file with dataset

    Returns:
        namedtuple format with fields:
        word1, first word
        word2, second word
        cntx1, context of first word
        cntx2, context of second word
        score, similarity score of pair
    """
    pairs = []
    with open(data, 'r') as infile:
        for line in infile:
            if '\t' in line:
                line = line.strip().split('\t')
            else:
                line = line.strip().split(' ')

            if line[2] == 'Human (mean)':
                continue
            pairs += [NonCntxPair(line[0].lower(), line[1].lower(), float(line[2]))]
    return pairs


def load_contextual(data):
    """
    Load non-contextual dataset.
    Args:
        data: file with dataset

    Returns:
        namedtuple format with fields:
        word1, first word
        word2, second word
        score, similarity score of pair
    """
    pairs = []
    with open(data, 'r') as infile:
        for line in infile:
            line = line.strip().split('\t')
            pairs += [CntxPair(line[1].lower(), line[3].lower(), line[5], line[6], float(line[7]))]
    return pairs


def matrix_cosine(m1, m2):
    """
    Estimate cosine similarities between 2 matrices.
    Args:
        m1 (numpy array): matrix 1
        m2 (numpy array): matrix 2

    Returns: (numpy array) cosine similarities
    """
    sim = cosine_similarity(m1, m2)
    return sim


def vector_cosine(v1, v2):
    """
    Estimate cosine similarity between 2 vectors:
    Args:
        v1 (numpy array): vector 1
        v2 (numpy array): vector 2

    Returns: semantic similarity
    """
    sim = 1 - distance.cosine(v1, v2)
    return sim


def normalize(embed):
    """
    Normalize a vector.
    Args:
        embed (numpy array): vector

    Returns: (numpy array) normalized vector
    """
    mag = np.sqrt(np.sum(embed ** 2))
    embed = embed / mag
    return np.array(embed)


def spearman_cor(truth, predict, ignore_zero=True):
    """
    Estimate Spearmanr correlation score between two lists of similarities.
    Association between the lists should be 1-by-1.
    Args:
        truth (list): ground truth scores
        predict (list): predicted scores
        ignore_zero (bool): ignore zero predicted similarities

    Returns:
        (int) number of evaluated pairs
        (float) spearman correlation score
    """
    if ignore_zero:
        truth_new = []
        predict_new = []
        for a_, b_ in zip(truth, predict):
            if b_ == 0:
                continue
            else:
                truth_new += [a_]
                predict_new += [b_]

        assert len(truth_new) == len(predict_new)
        return len(truth_new), spearmanr(truth_new, predict_new)[0]
    else:
        return len(truth), spearmanr(truth, predict)[0]
