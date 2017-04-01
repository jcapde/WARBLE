#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pickle
import argparse
import numpy as np

from collections import defaultdict
from gensim.models import LdaModel

def aggregate_by_keyword(corpus, dictionary):

    # Aggregate by keyword

    frequency = defaultdict(int)
    for doc in corpus:
        for word in doc:
            frequency[dictionary[word[0]]] += word[1]

    keywords = []
    minocurrences = 3
    for key, value in frequency.iteritems():
        if value > minocurrences:
            keywords.append(key)

    aggrTweets = defaultdict(list)
    for doc in corpus:
        for word in doc:
            if dictionary[word[0]] in keywords:
                aggrTweets[dictionary[word[0]]].extend([dictionary[word[0]] for word in doc])
    aggrCorpus = [vocabulary.doc2bow(text) for text in aggrTweets.values()]

    return aggrCorpus

def build_topic_matrix(LDA, T, vocabulary):

    M = len(vocabulary)
    topics = LDA.show_topics(T, M, formatted=False)
    Phi = np.zeros((T,M))
    for itopic, topic in enumerate(topics):
        for word in topic[1]:
            Phi[itopic, vocabulary.token2id[word[0]]] = word[1]
    return Phi

def get_theta(LDA, Corpus, T):
    pred_docs = LDA[Corpus]

    C = np.zeros((len(Corpus), T))
    for i, doc in enumerate(pred_docs):
        for topic in doc:
            C[i, topic[0]] = topic[1]

    return C

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Learn topics from tweets')
    parser.add_argument('-T', metavar='T', type=int, default=30)

    args = parser.parse_args()
    T = args.T

    vocabulary = pickle.load(open('data/tmp/vocabulary.pkl','rb'))
    or_corpus = pickle.load(open('data/tmp/corpus.pkl','rb'))
    corpus = aggregate_by_keyword(or_corpus, vocabulary)
    LDA = LdaModel(corpus, num_topics=T, id2word=vocabulary, passes=10)
    for topic in LDA.print_topics(num_topics=T, num_words=10):
        print topic
    Phi = build_topic_matrix(LDA, T, vocabulary)

    Theta = get_theta(LDA, or_corpus, T)

    pickle.dump(Phi, open('data/tmp/Phi.pkl', 'wb'))
    pickle.dump(Theta, open('data/tmp/Theta.pkl', 'wb'))


