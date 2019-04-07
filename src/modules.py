#!/usr/bin python3
# -*- coding : utf-8 -*
"""

Author: fenia
"""

from gensim import corpora, models
from tqdm import tqdm
import subprocess
from gensim.models import Word2Vec, KeyedVectors
import os
import logging
from collections import OrderedDict
from utils import normalize, vector_cosine
import pickle


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO, datefmt='%H:%M:%S')


class Corpus:
    def __init__(self, corpus_dir, doc_file, sents_file, stopwords):
        """
        Class tha constructs a corpus object.
        Args:
            corpus_dir (str): directory of the corpus
            doc_file (str): name of the corpus file in document per line format
            sents_file (str): name of the corpus file in sentence per line format
            stopwords (str): file with stopwords
        """
        self.stoplist = [i.strip() for i in open(stopwords, 'r')]

        self.doc_file = os.path.join(corpus_dir, doc_file)
        self.sents_file = os.path.join(corpus_dir, sents_file)

        self.dict_file = os.path.join(corpus_dir, doc_file+'.dict')
        self.txtdict_file = os.path.join(corpus_dir, doc_file+'.text_dict')
        self.bow_file = os.path.join(corpus_dir, doc_file+'.bow_mm')

    def build_crp(self, dic):
        """
        Args:
            dic: dictionary

        Returns: corpus in bag-of-words format
        """
        for line in open(self.doc_file, 'r'):
            yield dic.doc2bow(line.lower().split())

    def process(self):
        """
        Construction of dictionary, stopword removal, bag-of-words format.
        """
        logging.info('== Corpus Processing ===')

        # collect statistics for all tokens
        dictionary = corpora.Dictionary((line.lower().split() for line in open(self.doc_file, 'r')), prune_at=None)

        # remove stop words and words that appear only once
        stop_ids = [dictionary.token2id[stopword] for stopword in self.stoplist if stopword in dictionary.token2id]

        dictionary.filter_tokens(bad_ids=stop_ids)
        dictionary.filter_extremes(no_below=2, no_above=0.9)
        dictionary.compactify()
        dictionary.save(self.dict_file)
        dictionary.save_as_text(self.txtdict_file, sort_by_word=True)

        # corpus parsing
        bow_corpus = self.build_crp(dictionary)
        corpora.MmCorpus.serialize(self.bow_file, bow_corpus)

    def load_dict(self):
        """
        Load dictionary from file.
        """
        return corpora.Dictionary.load(self.dict_file)

    def load_corpus(self):
        """
        Load corpus from file.
        """
        return corpora.MmCorpus(self.bow_file)


class TopicModel:
    def __init__(self, lda_dir, topics, thres, topic_dir, corpus):
        """
        Class that constructs a topic model object.
        Args:
            lda_dir (str): directory to save LDA model
            topics (int): number of topics
            thres (float): probability threshold for construction of topic-corpora
            topic_dir (str): directory to save topic-corpora
            corpus (Corpus): corpus object to be used for training the topic model
        """
        self.corpus = corpus
        if os.path.isfile(corpus.dict_file):
            self.dic = corpus.load_dict()
        self.sent_corpus = corpus.sents_file

        self.topic_dir = topic_dir
        self.topics = topics
        self.thres = float(thres)
        self.model_file = os.path.join(lda_dir, str(topics)+'.lda')

    def build_model(self):
        """
        Build topic model using LDA.
        Saves model into file with extension '.lda'
        """
        self.dic = self.corpus.load_dict()
        bow_corpus = self.corpus.load_corpus()
        model = models.LdaMulticore(bow_corpus, id2word=self.dic, num_topics=self.topics, workers=3, iterations=200)
        model.save(self.model_file)

    def load_model(self):
        """
        Returns: Loaded LDA model from file.
        """
        return models.LdaModel.load(self.model_file, mmap='r')

    def build_topic_corpora(self):
        """
        Construction of topic-based corpora using posterior threshold.
        The corpus in sentence format is used (1 sentence per line)
        Save topic-corpora in separate files.
        """
        files = {}
        for topic in range(0, self.topics):
            files[topic] = open(os.path.join(self.topic_dir, str(topic)), 'w')

        total_ = subprocess.check_output("wc -l "+self.sent_corpus, shell=True).decode("utf-8").split(' ')[0]
        if self.thres == 1:
            logging.info('=== Clustering sentences to topic-based corpora: MAX criterion ===')
        else:
            logging.info('=== Clustering sentences to topic-based corpora: THRESHOLD {} ==='.format(self.thres))

        model = self.load_model()
        with open(self.sent_corpus, 'r') as in_corpus, tqdm(total=int(total_)) as pbar:

            for sent in in_corpus:
                pbar.update(1)

                if sent in ['\n', '\r\n']:
                    continue

                topic_preds = model[self.dic.doc2bow(sent.strip().split())]

                sent_topics = list(sorted(topic_preds, key=lambda x: x[1]))  # sort according to posteriors
                if not sent_topics:
                    return

                if self.thres == 1.0:
                    self.max_posterior(sent, sent_topics, files)
                else:
                    self.thres_posterior(sent, sent_topics, files)

        for topic in range(0, self.topics):
            files[topic].close()

    def max_posterior(self, sent, sent_topics, files):
        """
        Classifies sentences to the topic with the maximum LDA posterior.
        Args:
            sent (str): sentence
            sent_topics (list of tuples): topics where the sentence can be classified (topic, posterior)
            files (dict): files of topics to write sentence
        """
        max_topics = [st_[0] for st_ in sent_topics if st_[1] == sent_topics[-1][1]]  # topics with max posterior

        # make sure not all topics have same posterior --> garbage
        if len(max_topics) == self.topics:
            return

        for mt_ in max_topics:
            files[mt_].write(sent)  # write sentence

    def thres_posterior(self, sent, sent_topics, files):
        """
        Classify sentence to the topics with LDA posterior > threshold.
        Args:
            sent (str): sentence
            sent_topics (list of tuples): topics where the sentence can be classified (topic, posterior)
            files (dict): files of topics to write sentence
        """
        all_post = [st_[1] for st_ in sent_topics]

        # make sure not all topics have same posterior --> garbage
        if all_post[1:] == all_post[:-1]:
            return

        for st_ in sent_topics:
            if float(st_[1]) > float(self.thres):
                files[st_[0]].write(sent)


class DSM:
    def __init__(self, dsms_dir, corpus, name, size, window, cbow):
        """
        Class the constructs a Distributional-Semantic-Model (DSM) object.
        Args:
            dsms_dir (str): directory to save trained DSM
            corpus (str): corpus file
            name (str): name of the corpus
            size (int): dimensionality of vectors
            window (int): window size
            cbow (int): option for cbow or sg word2vec
        """
        self.bin_file = os.path.join(dsms_dir, name+'.bin')
        self.voc_file = os.path.join(dsms_dir, name+'.vocab')
        self.corpus = corpus

        self.size = str(size)
        self.win = str(window)
        self.cbow = str(cbow)

    def build_model(self):
        """
        Builds a DSM using Google's word2vec.
        """
        subprocess.call(['../google-w2v/word2vec',
                         '-train', self.corpus,
                         '-size', self.size,
                         '-window', self.win,
                         '-save-vocab', self.voc_file,
                         '-output', self.bin_file,
                         '-threads', str(7),
                         '-cbow', self.cbow,
                         '-binary', str(1)])

    def vectors(self):
        """
        Get semantic vectors stored in binary format.
        """
        wv_from_bin = KeyedVectors.load_word2vec_format(self.bin_file, binary=True)
        return wv_from_bin

    def extract_similarities(self, data, filename):
        """
        Extract similarities from trained DSM.
        Vectors are normalized to L2 norm before extracting the cosine similarity between them.
        Args:
            data (namedtuple): dataset for which to extract similarities
                               fields .word1, .word2 contain the word pair
            filename (str): filename to store similarities (pickle format)
        """
        dsm_vecs = self.vectors()
        dsm_vocab = dsm_vecs.wv.vocab

        similarities = OrderedDict()
        for d in data:
            if (d.word1 in dsm_vocab) and (d.word2 in dsm_vocab):
                similarities[(d.word1, d.word2)] = vector_cosine(normalize(dsm_vecs.wv.word_vec(d.word1)),
                                                                 normalize(dsm_vecs.wv.word_vec(d.word2)))
            else:
                print('Pair not in topic: {} - {}'.format(d.word1, d.word2))

        with open(filename + '.pkl', 'wb') as f:
            pickle.dump(similarities, f, pickle.HIGHEST_PROTOCOL)
        return similarities

    @staticmethod
    def load_similarities(filename):
        """
        Load similarities from pickle file.
        Arg:
            filename (str): filename with similarities
        Returns: dictionary with similarities for each word pair {(w1, w2): sim}
        """
        with open(filename + '.pkl', 'rb') as f:
            similarities = pickle.load(f)
        return similarities

