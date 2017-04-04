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
    label_df= DataFrame.from_csv('data/input/' + labelTweets, sep = ",")
    # Built labeled dataset with event and non-event tweets
    combined_df = df.merge(label_df, how="left", on="tweet_id")
    combined_df.tclass = combined_df.tclass.fillna("NonEvent")

    # Consider only labeled events with more than 10 tweets
    indices_eval = np.where(combined_df.groupby("tclass").count()["tweet_id"]<10)[0]
    for tclass in combined_df.groupby("tclass").count().ix[indices_eval,:].index:
        combined_df.ix[combined_df.tclass==tclass, "tclass"] = "NonEvent"

    return combined_df

def filter_by_day(dataset, day):
    th_l = day
    th_u = day + timedelta(days=1)
    dataset = dataset.ix[(dataset.time>th_l) & (dataset.time<=th_u),:]
    dataset.index = range(len(dataset))
    return dataset.sort_values("time")

def remove_user_duplicates(dataset):
    dataset = dataset.drop_duplicates("user_id")
    dataset.index = range(len(dataset))
    return dataset

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


def user_tweet_matrix(dataset):
    D = dataset.ix[:, ("tweet_id","user_id")]
    D.index = range(len(D))
    return D

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Preprocess tweets')
    parser.add_argument('-fileName', metavar='fileName', type=str, default='Twitter-DS_MERCE_2014_tweets.pkl')
    parser.add_argument('-labelTweets', metavar='labelTweets', type=str, default='Twitter-DS/MERCE/2014/labeled_events.csv')
    parser.add_argument('-day', metavar='day', type=str, default='24/09/2014')

    args = parser.parse_args()
    fileName = args.fileName
    labelTweets = args.labelTweets
    day = datetime.strptime(args.day,'%d/%m/%Y')

    # Build dataset
    dataset = build_dataset(fileName, labelTweets)
    print dataset.shape
    #print dataset.sort_values("time").tail()

    # Filter out tweets
    dataset = filter_by_day(dataset, day)
    print dataset.shape

    dataset = remove_user_duplicates(dataset)
    print dataset.shape

    # Transform spatio-temporal dimensions
    dataset, tn_mean, tn_std, ln_mean, ln_std = transform_space_time(dataset)
    print dataset.shape

    dataset = remove_space_outliers(dataset)
    print dataset.shape

    # Get cleaned text
    dataset = get_clean_text(dataset)
    print dataset.shape

    # Create corpus, remove words with low occurence and remove tweets without words
    vocabulary, corpus, dataset = get_corpus(dataset)
    w = create_word_matrix(vocabulary, corpus, dataset)
    print dataset.shape
    print w.shape

    # Create user - tweetID matrix for Tweet-SCAN
    user_tweet = user_tweet_matrix(dataset)

    #Create for tweets-SCAN
    pickle.dump([tn_mean, tn_std, ln_mean, ln_std], open('data/tmp/spacetime_stats.pkl', 'wb'))
    pickle.dump(w, open('data/tmp/w.pkl', 'wb'))
    pickle.dump(dataset, open('data/tmp/dataset.pkl','wb'))
    pickle.dump(vocabulary,open('data/tmp/vocabulary.pkl','wb'))
    pickle.dump(corpus,open('data/tmp/corpus.pkl','wb'))
    pickle.dump(user_tweet, open('data/tmp/user_tweet.pkl', 'wb'))

