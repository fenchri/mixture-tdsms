#!/usr/bin python3
# -*- coding : utf-8 -*
"""

Author: fenia
"""

import argparse
from modules import Corpus, TopicModel, DSM
from mixture import SemanticMixture as M
import utils
import os
import logging

parser = argparse.ArgumentParser()
parser.add_argument('--topics', type=int, default=50, help='Number of topics')
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
parser.add_argument('--dataset2', type=str, help='Second dataset given for Linear Regression if not cross validation')
parser.add_argument('--name', type=str, help='Dataset name')
parser.add_argument('--construct', action='store_true', help='Create topic-based corpora & DSMs')
parser.add_argument('--sim_load', action='store_true', help='Load semantic similarities from files.')
args = parser.parse_args()


# Create output directories
if args.cbow == 1:
    temp = 'cbow_d' + str(args.size) + '_w' + str(args.window)
else:
    temp = 'sg_d' + str(args.size) + '_w' + str(args.window)
dsms_dir = os.path.join(args.save_dir, 'dsms')

if not os.path.exists(dsms_dir):
    os.makedirs(dsms_dir)

tdsms_dir = os.path.join(dsms_dir, str(args.topics), temp)
if not os.path.exists(tdsms_dir):
    os.makedirs(tdsms_dir)

lda_dir = os.path.join(args.save_dir, 'lda')
if not os.path.exists(lda_dir):
    os.makedirs(lda_dir)

topic_dir = os.path.join(args.save_dir, 'topic-corpora', str(args.topics))
if not os.path.exists(topic_dir):
    os.makedirs(topic_dir)

topic_sims_dir = os.path.join(args.save_dir, 'sims', temp, str(args.topics), args.name)
if not os.path.exists(topic_sims_dir):
    os.makedirs(topic_sims_dir)


corpus = Corpus(args.corpus_dir, args.docs, args.sents, args.stopwords)
topic_model = TopicModel(lda_dir, args.topics, args.thres, topic_dir, corpus)

topic_dsms = {}
for t_ in range(args.topics):
    topic_dsms[t_] = DSM(tdsms_dir, os.path.join(topic_dir, str(t_)), str(t_), args.size, args.window, args.cbow)

if args.construct:
    # Corpus processing
    corpus.process()

    # Topic Modeling
    topic_model.build_model()
    topic_model.build_topic_corpora()

    # Build DSMS
    logging.info("=== Building Topic-based Distributional Semantic Models ===")  # topic-based models
    for t_ in range(args.topics):
        topic_dsms[t_].build_model()


# Semantic Mixture Model
if args.context:
    # CONTEXTUAL
    pairs = utils.load_contextual(args.dataset)
    ground_truth = [p.score for p in pairs]

    mix = M(pairs, topic_dsms, args.topics, topic_model, topic_sims_dir, load=args.sim_load)
    scores = {'MaxSimC': utils.spearman_cor(truth=ground_truth, predict=[mix.MaxSimC(p) for p in pairs]),
              'AvgSim': utils.spearman_cor(truth=ground_truth, predict=[mix.AvgSim(p) for p in pairs]),
              'AvgSimC': utils.spearman_cor(truth=ground_truth, predict=[mix.AvgSimC(p) for p in pairs])}

    print('\n=== Results for {} Topics ==='.format(args.topics))
    for key, val in scores.items():
        print('# Pairs {}/{}\t{:<10}\tSpearmanr = {:.4f}'.format(val[0], len(pairs), key, val[1]))


else:
    # NON-CONTEXTUAL
    pairs = utils.load_non_contextual(args.dataset)
    ground_truth = [p.score for p in pairs]

    if args.dataset2:
        tmp = utils.load_non_contextual(args.dataset2)
        mix2 = M(pairs+tmp, topic_dsms, args.topics, topic_model, topic_sims_dir, load=args.sim_load)
        scores = {'MaxSim': utils.spearman_cor(truth=ground_truth, predict=[mix2.MaxSim(p) for p in pairs]),
                  'AvgSim': utils.spearman_cor(truth=ground_truth, predict=[mix2.AvgSim(p) for p in pairs]),
                  'LR': mix2.lr_train(tmp, pairs, max([p.score for p in tmp]))}
    else:
        mix = M(pairs, topic_dsms, args.topics, topic_model, topic_sims_dir, load=args.sim_load)
        scores = {'MaxSim': utils.spearman_cor(truth=ground_truth, predict=[mix.MaxSim(p) for p in pairs]),
                  'AvgSim': utils.spearman_cor(truth=ground_truth, predict=[mix.AvgSim(p) for p in pairs]),
                  'LR': mix.lr_cross_val(pairs, max([p.score for p in pairs]), folds=3)}

    print('\n=== Results for {} Topics ==='.format(args.topics))
    for key, val in scores.items():
        print('# Pairs {}/{}\t{:<10}\tSpearmanr = {:.4f}'.format(val[0], len(pairs), key, val[1]))

