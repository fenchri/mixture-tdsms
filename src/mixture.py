#!/usr/bin python3
# -*- coding : utf-8 -*
"""

Author: fenia
"""

import os
import numpy as np
import re
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
import utils
from random import shuffle


class SemanticMixture:
    def __init__(self, pairs, tdsms, topics, topic_model, topic_sims_dir, load=False):
        """
        Class that constructs a mixture object for a given dataset.
        Args:
            tdsms (dict of DSM): topic-DSMs
            topics (int): number of topics
            topic_model (TopicModel): trained topic model
        """
        self.topics = topics
        self.tdsms = {}

        for t_ in range(topics):
            if load:
                self.tdsms[t_] = tdsms[t_].load_similarities(os.path.join(topic_sims_dir, str(t_)))
            else:
                self.tdsms[t_] = tdsms[t_].extract_similarities(pairs, os.path.join(topic_sims_dir, str(t_)))

        self.topic_model = topic_model.load_model()
        self.topic_model.minimum_phi_value = 0.01
        self.topic_model.per_word_topics = False
        self.dic = topic_model.dic

    def AvgSim(self, pair):
        """
        Average Similarity score.
        All topic-based similarities contribute equally to the final similarity score.
        Args:
            pair (namedtuple): a word pair

        Returns: (float) similarity score
        """
        total_sim = []
        for t_ in range(self.topics):
            if (pair.word1, pair.word2) in self.tdsms[t_]:
                total_sim += [self.tdsms[t_][(pair.word1, pair.word2)]]
        return np.mean(total_sim) if total_sim else 0

    def MaxSim(self, pair):
        """
        Maximum Similarity score.
        The maximum similarity score from all topic-based similarities is considered as the final similarity score.
        Args:
            pair (namedtuple): a word pair

        Returns: (float) similarity score
        """
        total_sim = []
        for t_ in range(self.topics):
            if (pair.word1, pair.word2) in self.tdsms[t_]:
                total_sim += [self.tdsms[t_][(pair.word1, pair.word2)]]
        return np.max(total_sim) if total_sim else 0

    def AvgSimC(self, pair):
        """
        Average Similarity score for contextual datasets.
        Assuming we know the context were each word of the pair resides,
        a weighted average of topic-based similarities is estimated using the topic posterior
        probabilities as weights (based on a trained LDA model).
        Args:
            pair (namedtuple): a word pair

        Returns: (float) similarity score
        """
        cntx = pair.cntx1 + pair.cntx2
        cntx = cntx.replace('<b>', '')
        cntx = cntx.replace('</b>', '')
        cntx = re.sub(' +', ' ', cntx)

        tops_n_posts = self.topic_model[self.dic.doc2bow(cntx.lower().split(' '))]

        posts = 0
        sims = 0
        for topic_, post_ in tops_n_posts:

            if (pair.word1, pair.word2) in self.tdsms[topic_]:
                sims += post_ * self.tdsms[topic_][(pair.word1, pair.word2)]
                posts += post_
        return sims/posts if posts != 0 else 0

    def MaxSimC(self, pair):
        """
        Maximum Similarity score for contextual datasets.
        Assuming we know the context were each word of the pair resides,
        the similarity score of the pair is estimated from the topic with the maximum posterior probability,
        (based on a trained LDA topic model).
        Args:
            pair (namedtuple): a word pair

        Returns: (float) similarity score
        """
        cntx = pair.cntx1 + pair.cntx2
        cntx = cntx.replace('<b>', '')
        cntx = cntx.replace('</b>', '')
        cntx = re.sub(' +', ' ', cntx)

        tops_n_posts = self.topic_model[self.dic.doc2bow(cntx.lower().split(' '))]

        max_topic = sorted(tops_n_posts, key=lambda tup: tup[1])[-1]
        tdsm = self.tdsms[max_topic[0]]

        sim = 0
        if (pair.word1, pair.word2) in tdsm:
            sim = tdsm[(pair.word1, pair.word2)]
        return sim

    def LR(self, train_pairs, test_pairs, max_range):
        """
        Linear Regression. Learn topic-based weights.
        Args:
            train_pairs: pairs for training
            test_pairs: pairs for testing
            max_range: maximum similarity score in the dataset

        Returns: (numpy array) similarities for the test pairs
        """
        train_mat = np.empty([len(train_pairs), self.topics], dtype=float)
        test_mat = np.empty([len(train_pairs), ], dtype=float)
        pred_mat = np.empty([len(test_pairs), self.topics], dtype=float)

        for t_ in range(self.topics):

            # train pairs
            for i, ptr in enumerate(train_pairs):
                if (ptr.word1, ptr.word2) in self.tdsms[t_]:
                    train_mat[i, t_] = self.tdsms[t_][(ptr.word1, ptr.word2)]
                else:
                    train_mat[i, t_] = 0

            # prediction pairs
            for j, pts in enumerate(test_pairs):
                if (pts.word1, pts.word2) in self.tdsms[t_]:
                    pred_mat[j, t_] = self.tdsms[t_][(pts.word1, pts.word2)]
                else:
                    pred_mat[j, t_] = 0

        for k, p in enumerate(train_pairs):
            test_mat[k] = float(p.score) / float(max_range)  # normalize to [0, 1] ground truth

        reg = LinearRegression().fit(train_mat, test_mat)
        sum_ = sum(reg.coef_) + reg.intercept_

        final_scores = (reg.intercept_ / sum_) + np.matmul((reg.coef_ / sum_), pred_mat.T)
        return final_scores

    def lr_cross_val(self, pairs, max_range, folds=3):
        """
        Cross validation for Linear Regression
        Args:
            pairs: list of namedtuple pairs
            max_range: maximum similarity score on the dataset
            folds: Number of folds

        Returns: (float) average of spearman correlation over the folds
        """
        kf = KFold(n_splits=folds, shuffle=True)

        scores = []
        for train_index, test_index in kf.split(pairs):
            x_train, x_test = [pairs[k] for k in train_index], [pairs[m] for m in test_index]

            result = self.LR(x_train, x_test, max_range).ravel().tolist()

            num, scr = utils.spearman_cor(truth=[x.score for x in x_test], predict=result)
            scores += [scr]
        return num, np.mean(scores)

    def lr_train(self, train_pairs, pairs, max_range):
        """
        Simple training for Linear Regression using a training dataset.
        Args:
            pairs: list of namedtuple pairs for testing
            train_pairs: list of namedtuple pairs for training
            max_range: maximum similarity score on the dataset

        Returns: (float) spearman correlation
        """
        shuffle(train_pairs)
        result = self.LR(train_pairs, pairs, max_range).ravel().tolist()

        num, score = utils.spearman_cor(truth=[x.score for x in pairs], predict=result)
        return num, score







