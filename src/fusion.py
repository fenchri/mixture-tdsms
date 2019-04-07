#!/usr/bin python3
# -*- coding : utf-8 -*
"""

Author: fenia
"""

import os
from mixture import SemanticMixture as M
from modules import Corpus, TopicModel, DSM
import argparse
import utils
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--topics', type=int, nargs='*', help='Groups of topics to fuse')
    parser.add_argument('--stopwords', type=str, default='../data/stopwords.txt', help='File with stopwords')
    parser.add_argument('--corpus_dir', type=str, help='Directory with corpus files')
    parser.add_argument('--docs', type=str, help='File with corpus in document format')
    parser.add_argument('--sents', type=str, help='File with corpus in sentence format')
    parser.add_argument('--size', type=int, default=300, help='Vector dimensionality for the word2vec models')
    parser.add_argument('--window', type=int, default=5, help='the window size for the word2vec models')
    parser.add_argument('--cbow', type=int, default=1, choices=[0, 1], help='Word2vec model (skip-gram=0, cbow=1)')
    parser.add_argument('--thres', type=float, default=0.1, help='Threshold for topic-corpora creation. If thres=1 '
                                                                 'the topic with the maximum posterior os used')
    parser.add_argument('--save_dir', type=str, help='Output directory for models saving')
    parser.add_argument('--context', action='store_true', help='Contextual/non-contextual dataset')
    parser.add_argument('--dataset', type=str, help='Semantic dataset to test model')
    parser.add_argument('--name', type=str, help='Dataset name')
    args = parser.parse_args()

    # init
    if args.cbow == 1:
        temp = 'cbow_d' + str(args.size) + '_w' + str(args.window)
    else:
        temp = 'sg_d' + str(args.size) + '_w' + str(args.window)
    dsms_dir = os.path.join(args.save_dir, 'dsms')
    lda_dir = os.path.join(args.save_dir, 'lda')

    corpus = Corpus(args.corpus_dir, args.docs, args.sents, args.stopwords)
    pairs = utils.load_contextual(args.dataset)
    ground_truth = [p.score for p in pairs]

    scores = {}
    for tg in args.topics:
        tdsms_dir = os.path.join(dsms_dir, str(tg), temp)
        topic_sims_dir = os.path.join(args.save_dir, 'sims', temp, str(tg), args.name)
        topic_dir = os.path.join(args.save_dir, 'topic-corpora', str(args.thres) + '_thres', str(tg))
        topic_model = TopicModel(lda_dir, tg, args.thres, topic_dir, corpus)

        topic_dsms = {}
        for t_ in range(tg):
            topic_dsms[t_] = DSM(tdsms_dir, os.path.join(topic_dir, str(t_)), str(t_), args.size, args.window, args.cbow)

        mix = M(pairs, topic_dsms, tg, topic_model, topic_sims_dir, load=True)

        scores[tg] = {'MaxSimC': [mix.MaxSimC(p) for p in pairs],
                      'AvgSimC': [mix.AvgSimC(p) for p in pairs],
                      'AvgSim': [mix.AvgSim(p) for p in pairs]}

    mat_maxc = np.empty([len(args.topics), len(pairs)], dtype=float)
    mat_avgc = np.empty([len(args.topics), len(pairs)], dtype=float)
    mat_avg = np.empty([len(args.topics), len(pairs)], dtype=float)
    for i, tg in enumerate(scores):
        mat_maxc[i] = scores[tg]['MaxSimC']
        mat_avgc[i] = scores[tg]['AvgSimC']
        mat_avg[i] = scores[tg]['AvgSim']

    fin_maxc = np.max(mat_maxc, axis=0)
    fin_avgc = np.max(mat_avgc, axis=0)
    fin_avg = np.max(mat_avg, axis=0)

    print('\n=== Results for fusion of {} Topics ==='.format(args.topics))
    num, score = utils.spearman_cor(truth=ground_truth, predict=fin_maxc.ravel().tolist())
    print('# Pairs {}/{}\tMaxSimC\tSpearmanr = {:.4f}'.format(num, len(pairs), score))

    num, score = utils.spearman_cor(truth=ground_truth, predict=fin_avg.ravel().tolist())
    print('# Pairs {}/{}\tAvgSim \tSpearmanr = {:.4f}'.format(num, len(pairs), score))

    num, score = utils.spearman_cor(truth=ground_truth, predict=fin_avgc.ravel().tolist())
    print('# Pairs {}/{}\tAvgSimC\tSpearmanr = {:.4f}'.format(num, len(pairs), score))


if __name__ == '__main__':
    main()






