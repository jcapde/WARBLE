#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pickle
import argparse

from collections import defaultdict
from datetime import datetime, timedelta
import numpy as np
import pyproj
import re
from gensim.corpora import Dictionary

from nltk.corpus import stopwords
from pandas import DataFrame, read_pickle



def build_dataset(fileName, labelTweets):
    # Load raw tweets
    df = read_pickle('data/input/' + fileName)
    # Load labels
    label_df= DataFrame.from_csv('data/input/' + labelTweets, sep = ",", index_col=None)
    # Built labeled dataset with event and non-event tweets
    combined_df = df.merge(label_df, how="left", on="tweet_id")
    combined_df.tclass = combined_df.tclass.fillna("NonEvent")
    return combined_df



def filter_by_day(tmpdf, day):
    th_l = day
    th_u = day + timedelta(days=1)
    tmpdf = tmpdf.ix[(tmpdf.time>th_l) & (tmpdf.time<=th_u),:]
    tmpdf.index = range(len(tmpdf))
    return tmpdf

def remove_user_duplicates(tmpdf):
    tmpdf = tmpdf.drop_duplicates("user_id")
    tmpdf.index = range(len(tmpdf))
    return tmpdf

def transform_space_time(dataset):

    # Project spatial coordinates to UTM
    # Project temporal dimension to seconds
    UTM31 = pyproj.Proj("+init=EPSG:32631")
    wgs84 = pyproj.Proj("+init=EPSG:4326")
    ln = np.zeros(shape=(dataset.shape[0],2))
    tn = np.zeros(dataset.shape[0])
    auxmin = min(dataset["time"])
    for n, ind in enumerate(dataset.index):
        ln[n, :] = pyproj.transform(wgs84,UTM31, *dataset.loc[ind,["coordx","coordy"]])
        tn[n] = (dataset.loc[ind,"time"]-auxmin).total_seconds()

    # Normalize spatio-temporal dimensions

    ln_mean = np.mean(ln, axis=0)
    ln_std = np.std(ln, axis=0)
    ln = (ln-ln_mean)/ln_std

    dataset['x'] = ln[:,0]
    dataset['y'] = ln[:,1]

    tn_mean = np.mean(tn)
    tn_std = np.std(tn)
    tn = (tn-tn_mean)/tn_std
    dataset['t'] = tn

    return dataset, tn_mean, tn_std, ln_mean, ln_std

def remove_space_outliers(dataset):

    # Remove spatial outliers

    indices = np.where(np.linalg.norm(dataset.as_matrix(["x", "y"]),axis=1)<4)[0]
    dataset = dataset.ix[indices,:]
    dataset.index = range(len(dataset.index))
    return dataset

def get_clean_text(dataset):

    highpoints = re.compile(u'[\uD800-\uDBFF][\uDC00-\uDFFF]')
    pre = re.compile(r'[^\w\s]',re.UNICODE)
    stoplist = [u'barcelona',u'bcn',u'merce',u'lamerce',u'merc\xe8',u'lamerc\xe8']
    stoplist.extend(stopwords.words('english'))
    stoplist.extend(stopwords.words('spanish'))
    stoplist.extend(stopwords.words('catalan'))

    # Get cleaned tweets
    cltext = []
    for hit in dataset.text:
        # remove urls
        tweet = re.sub(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))', '', hit)
        # remove emojii
        tweet = highpoints.sub('', tweet)
        # remove numbers
        tweet = re.sub("\d+", "", tweet)
        # remove mentions and the hash sign
        tweet = re.sub("(@[A-Za-z0-9]+)"," ",tweet).lower()
        tweet = re.sub("#", "", tweet)
        tweet = re.sub("_", "", tweet)
        tweet = pre.sub("",tweet)
        cltext.append(" ".join([i for i in tweet.split() if ((i not in stoplist))]))

    dataset["cltext"] = cltext
    return dataset

def get_corpus(dataset):

    # Word counts
    frequency = defaultdict(int)
    for text in dataset.cltext.values:
        for token in text.split(" "):
            frequency[token] += 1

    # Remove words with zero and one counts
    tweets = [[token for token in text.split(" ") if frequency[token] > 1] for text in dataset.cltext.values]

    # Remove tweets with no words
    dictionary = Dictionary(tweets)
    tweetCorpus = [dictionary.doc2bow(text) for text in tweets]
    tweetIndex = [auxdoc for auxdoc in xrange(len(tweetCorpus)) if len(tweetCorpus[auxdoc])>0]
    Tweets = [tex for t, tex in enumerate(tweets) if t in tweetIndex]

    #Regenerate corpus and dataset
    Corpus = [dictionary.doc2bow(text) for text in Tweets]
    dataset = dataset.ix[tweetIndex,:]

    return dictionary, Corpus, dataset

def create_word_matrix(vocabulary, corpus, dataset):

    N = len(dataset)
    M = len(vocabulary)
    wnm = np.zeros((N,M))
    for n, doc in enumerate(corpus):
        for word in doc:
            wnm[n, word[0]] += word[1]

    return wnm


def remove_labels(dataset):

    # Consider only labeled events with more than 10 tweets
    indices_eval = np.where(dataset.groupby("tclass").count()["tweet_id"]<10)[0]
    for tclass in dataset.groupby("tclass").count().ix[indices_eval,:].index:
        dataset.ix[dataset.tclass==tclass, "tclass"] = "NonEvent"

    return dataset
        
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Preprocess tweets')
    parser.add_argument('-fileName', metavar='fileName', type=str, default='All_Twitter-DS_MERCE_2014_tweets.pkl')
    parser.add_argument('-labelTweets', metavar='labelTweets', type=str, default='Twitter-DS/MERCE/2014/labeled_events_day_24.csv')
    parser.add_argument('-day', metavar='day', type=str, default='24/09/2014')

    args = parser.parse_args()
    fileName = args.fileName
    labelTweets = args.labelTweets
    day = datetime.strptime(args.day,'%d/%m/%Y')

    # Build dataset
    dataset = build_dataset(fileName, labelTweets)
    print "Initial number of tweets: " + str(dataset.shape[0])

    # Filter out tweets
    dataset = filter_by_day(dataset, day)
    print "Filtered tweets from " + str(args.day) +": "+ str(dataset.shape[0])

    dataset = remove_user_duplicates(dataset)

    # Transform spatio-temporal dimensions
    dataset, tn_mean, tn_std, ln_mean, ln_std = transform_space_time(dataset)
    dataset = remove_space_outliers(dataset)

    # Get cleaned text
    dataset = get_clean_text(dataset)
    print "Final list of cleaned tweets: " + str(dataset.shape[0])

    # Create corpus, remove words with low occurence and remove tweets without words
    vocabulary, corpus, dataset = get_corpus(dataset)
    w = create_word_matrix(vocabulary, corpus, dataset)

    # Remove label from events with less than 10 tweets
    dataset = remove_labels(dataset)

    #Create for tweets-SCAN
    pickle.dump([tn_mean, tn_std, ln_mean, ln_std], open('data/tmp/spacetime_stats.pkl', 'wb'))
    pickle.dump(w, open('data/tmp/w.pkl', 'wb'))
    pickle.dump(dataset, open('data/tmp/dataset.pkl','wb'))
    pickle.dump(vocabulary,open('data/tmp/vocabulary.pkl','wb'))
    pickle.dump(corpus,open('data/tmp/corpus.pkl','wb'))

