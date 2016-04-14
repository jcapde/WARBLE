#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from pandas import DataFrame, read_pickle
import matplotlib.pyplot as plt
import pyproj
from datetime import datetime
import scipy.fftpack
import scipy.ndimage
from gensim.models import LdaModel
from gensim.corpora import Dictionary
from nltk.corpus import stopwords
from collections import defaultdict
import pyproj, re
import sys
import pickle

if __name__ == "__main__":

    # Load tweets
    df = read_pickle('./data/input/full_dataset.pkl')
    dftagged = DataFrame.from_csv('./data/input/labeled_events.csv', sep =";")

    dfcomp = df.merge(dftagged, how="left", on="tweet_id")
    dfcomp.tclass = dfcomp.tclass.fillna("-- Background")

    # Select one day tweets
    i = 24
    th_l = datetime.strptime('Sep ' + str(i) + ' 2014  12:00AM', '%b %d %Y %I:%M%p')
    th_u = datetime.strptime('Sep ' + str(i+1) + ' 2014  12:00AM', '%b %d %Y %I:%M%p')
    dfcomp = dfcomp.ix[(dfcomp.time>th_l) & (dfcomp.time<=th_u),:]
    dfcomp.index = range(len(dfcomp))

    # Select unique users
    dfcomp = dfcomp.drop_duplicates("user_id")
    dfcomp.index = range(len(dfcomp))

    # Project spatial coordinates to UTM
    # Project temporal dimension to seconds
    UTM31 = pyproj.Proj("+init=EPSG:32631")
    wgs84 = pyproj.Proj("+init=EPSG:4326")
    ln = np.zeros(shape=(dfcomp.shape[0],2))
    tn = np.zeros(dfcomp.shape[0])
    auxmin = min(dfcomp["time"])
    for n, ind in enumerate(dfcomp.index):
        timestr = '\rSpatio-temporal transformations: %%%i\t' % int(n*1./len(dfcomp.index)*100)
        sys.stdout.write(timestr)
        sys.stdout.flush()
        ln[n, :] = pyproj.transform(wgs84,UTM31, *dfcomp.loc[ind,["coordx","coordy"]])
        tn[n] = (dfcomp.loc[ind,"time"]-auxmin).total_seconds()

    timestr = '\rSpatio-temporal transformations: %%%i\t' % 100
    sys.stdout.write(timestr)
    sys.stdout.flush()
    print "  "

    # Normalize spatio-temporal dimensions
    ln_mean = np.mean(ln, axis=0)
    ln_std = np.std(ln, axis=0)
    ln = (ln-ln_mean)/ln_std
    tn_mean = np.mean(tn)
    tn_std = np.std(tn)
    tn = (tn-tn_mean)/tn_std

    # Remove spatial outliers
    indices = np.where(np.linalg.norm(ln,axis=1)<4)[0]
    ln = ln[indices,:]
    tn = tn[indices]
    dfcomp = dfcomp.ix[indices,:]
    dfcomp.index = range(len(dfcomp))

    # Define text stoplist and unicode transformations
    highpoints = re.compile(u'[\uD800-\uDBFF][\uDC00-\uDFFF]')
    pre = re.compile(r'[^\w\s]',re.UNICODE)
    stoplist = [u'barcelona',u'bcn',u'merce',u'lamerce',u'merc\xe8',u'lamerc\xe8']
    stoplist.extend(stopwords.words('english'))
    stoplist.extend(stopwords.words('spanish'))
    stoplist.extend(stopwords.words('catalan'))

    # Get cleaned tweets
    rawTweets = []
    for hit in dfcomp.text:
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
        rawTweets.append([i for i in tweet.split() if ((i not in stoplist))])

    # Word count
    frequency = defaultdict(int)
    for text in rawTweets:
        for token in text:
            frequency[token] += 1
    tweets = [[token for token in text if frequency[token] > 1] for text in rawTweets]


    # Remove tweets with no words
    dictionary = Dictionary(tweets)
    tweetCorpus = [dictionary.doc2bow(text) for text in tweets]
    tweetIndex = [auxdoc for auxdoc in xrange(len(tweetCorpus)) if len(tweetCorpus[auxdoc])>0]
    Tweets = [tex for t, tex in enumerate(tweets) if t in tweetIndex]
    Corpus = [dictionary.doc2bow(text) for text in Tweets]

    # Final data
    dfcomp = dfcomp.ix[tweetIndex,:]

    ln = ln[tweetIndex,:]
    tn = tn[tweetIndex]
    N = len(dfcomp)
    M = len(dictionary)
    wnm = np.zeros((N,M))
    for n, doc in enumerate(Corpus):
        for word in doc:
            wnm[n, word[0]] += word[1]

    # Phi Blei Model
    keywords = []
    minocurrences = 3
    for key, value in frequency.iteritems():
        if value > minocurrences:
            keywords.append(key)

    aggrTweets = defaultdict(list)
    for doc in Tweets:
        for word in doc:
            if word in keywords:
                aggrTweets[word].extend(doc)

    aggrCorpus = [dictionary.doc2bow(text) for text in aggrTweets.values()]

    T = 30
    M = len(dictionary)
    modelLDA = LdaModel(aggrCorpus, num_topics=T, id2word=dictionary, passes=10)
    topics = modelLDA.show_topics(T, M, formatted=False)
    Phi = np.zeros((T,M))
    for itopic, topic in enumerate(topics):
        for word in topic:
            Phi[itopic, dictionary.token2id[word[1]]] = word[0]

    # Background models
    avg_time = np.zeros(1)
    avg_loc = np.zeros((1,2))
    for i in [19, 20, 21, 22, 23]:
        th_l = datetime.strptime('Sep '+ str(i) + ' 2014  12:00AM', '%b %d %Y %I:%M%p')
        th_u = datetime.strptime('Sep ' + str(i+1) + ' 2014  12:00AM', '%b %d %Y %I:%M%p')
        selection = df.ix[(df.time > th_l) & (df.time <= th_u),:]
        length = selection.shape[0]
        selection.index = xrange(length)
        aux = np.zeros(shape=(length,2))
        for n in xrange(selection.shape[0]):
            aux[n, :] = pyproj.transform(wgs84, UTM31, *selection.ix[n, ["coordx","coordy"]])
        aux_loc = np.copy((aux-ln_mean)/ln_std)
        aux_time = (np.array([delta.total_seconds() for delta in selection["time"] - np.min(selection["time"])])-tn_mean)/tn_std
        mask = np.ones(aux_loc.shape[0], dtype=bool)
        if len(mask)!=0:
            mask[np.argmin(aux_loc,axis=0)[0]] = False
            aux_loc = np.copy(aux_loc[mask,:])
            aux_time = np.copy(aux_time[mask])
        avg_time = np.append(avg_time, aux_time)
        avg_loc = np.append(avg_loc, aux_loc, axis=0)

    # Temporal FFT-IFFT
    values, bins, patch = plt.hist(avg_time[1:], bins = 100, normed=True)
    w = scipy.fftpack.rfft(values)
    f = scipy.fftpack.rfftfreq(len(values), bins[1]-bins[0])
    spectrum = w**2
    cutoff_idx = f > 0.6
    w2 = w.copy()
    w2[cutoff_idx] = 0
    y2 = scipy.fftpack.irfft(w2)
    values = np.copy(y2)

    # Spatial Gaussian filter
    H, xedges, yedges = np.histogram2d(avg_loc[1:,0], avg_loc[1:,1], bins=(40, 40), normed = True)
    H = np.copy(H.T)
    im = plt.imshow(H, interpolation='nearest', origin='low', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
    plt.show()
    Hout = scipy.ndimage.gaussian_filter(H, 1.5)
    H = np.copy(Hout)
    np.abs(xedges[:len(xedges)-1]-xedges[1:])[0]*np.abs(yedges[:len(yedges)-1]-yedges[1:])[0]*np.sum(H)
    im = plt.imshow(H, interpolation='nearest', origin='low', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
    plt.show()
    H = np.copy(H.T)

    ## Consider only tagged events with more than 10 tweets
    indices_eval = np.where(dfcomp.groupby("tclass").count()["tweet_id"]<10)[0]
    for tclass in dfcomp.groupby("tclass").count().ix[indices_eval,:].index:
        dfcomp.ix[dfcomp.tclass==tclass,"tclass"] = "-- Background"

    lab = np.zeros(N)
    for i, c in enumerate(list(set(dfcomp.tclass))):
        lab[np.where(dfcomp.tclass==c)[0]] = i

    pred_docs = modelLDA[Corpus]

    C = np.zeros((len(Corpus), T))
    for i, doc in enumerate(pred_docs):
        for topic in doc:
            C[i, topic[0]] = topic[1]

    D = dfcomp.ix[:, ("tweet_id","user_id")]
    D.index = range(len(D))


    np.save('data/input/ln.npy', ln)
    np.save('data/input/tn.npy', tn)
    np.save('data/input/ln_std.npy', ln_std)
    np.save('data/input/tn_std.npy', tn_std)
    np.save('data/input/wnm.npy', wnm)
    np.save('data/input/Phi.npy', Phi)
    np.save('data/input/label.npy', lab)
    np.save('data/input/xedges.npy', xedges)
    np.save('data/input/yedges.npy', yedges)
    np.save('data/input/H.npy', H)
    np.save('data/input/bins.npy', bins)
    np.save('data/input/values.npy', values)
    pickle.dump(dfcomp, open('data/input/dfcomp.pkl','wb'))
    np.save('data/input/C.npy', C)
    pickle.dump(D, open('data/input/D.pkl', 'wb'))
    pickle.dump(dictionary,open('data/input/dictionary.pkl','wb'))
    pickle.dump(Tweets,open('data/input/Tweets.pkl','wb'))

