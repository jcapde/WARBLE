#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pickle
import argparse
import pyproj
import math
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
import functions.ev_functions_BCubed

from models.WARBLE import inference_wback_summary, inference_wback
from functions.ev_functions import evaluate_recall_event, map_estimate, evaluate_wback

from mpl_toolkits.mplot3d import Axes3D

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Learn topics from tweets')
    parser.add_argument('-T', metavar='T', type=int, default=30)
    parser.add_argument('-K', metavar='K', type=int, default=8)
    parser.add_argument('-maxIter', metavar='maxIter', type=int, default=50)
    parser.add_argument('-day', metavar='day', type=str, default='24/09/2014')

    args = parser.parse_args()
    day = datetime.strptime(args.day,'%d/%m/%Y')


    ## LOAD DATA
    # Dataset
    dataset = pickle.load(open('data/tmp/dataset.pkl','rb'))
    # Word Matrix
    w = pickle.load(open('data/tmp/w.pkl', 'rb'))
    # Space time stats
    tn_mean, tn_std, ln_mean, ln_std = pickle.load(open('data/tmp/spacetime_stats.pkl', 'rb'))
    # Background
    xs, ys, HistLoc, ts, HistTemp = pickle.load(open('data/tmp/background.pkl','rb'))

    vocabulary = pickle.load(open('data/tmp/vocabulary.pkl', 'rb'))


    ## INITIALIZE VARIABLES

    N, M = w.shape
    T = args.T
    K = args.K
    maxIter = args.maxIter

    ## HYPERPARAMETERS
    # Proportions
    a_pi = np.array([100.]*K)

    # Space hyperparameters
    ln = dataset.as_matrix(["x", "y"])
    # Normal hyperparameters
    m_mu = np.mean(ln, axis=0)
    beta_mu = 9./(10*np.linalg.norm(ln.max(axis=0)-ln.min(axis=0))**2)
    # Inverse Wishart hyperparameters
    W_Delta = 10*np.eye(2)
    nu_Delta = 100.

    # Time hyperparameters
    tn = dataset.as_matrix(["t"])[:,0]
    # Normal hyperparameters
    m_tau = np.mean(tn)
    beta_tau = 9/(max(tn)-min(tn))**2
    # Inverse Wishart hyperparameters
    a_lambda = 100.
    b_lambda = 1.

    # Event-Topic proportions hyperparameter
    a_theta = 0.1

    # Topic-Word proportions hyperparameter
    a_phi = 0.1


    ## Run models
    event_assig = np.zeros(N)
    print("WARBLE model ----------------")
    en_, thetak_, phik_, mk_, betamuk_, nuk_, Wk_, mtauk_, betatauk_, ak_, bk_ = inference_wback_summary(maxIter, tn, ln, w, a_pi, T, K, m_mu, beta_mu, W_Delta, nu_Delta, m_tau, beta_tau, a_lambda, b_lambda, a_theta, a_phi, HistTemp, ts, HistLoc, xs, ys)

    pickle.dump([en_, thetak_, phik_, mk_, betamuk_, nuk_, Wk_, mtauk_, betatauk_, ak_, bk_], open('data/output/WARBLEsummary.npy', 'wb'))
    event_assig = map_estimate(en_)

    pr, re, f = evaluate_wback(dataset, event_assig)
    print "WARBLE model -", " Purity: ", pr, " Inv. Purity: ", re, " F-measure:", f

    #print dataset.groupby("tclass").count()["tweet_id"]
    recalls = evaluate_recall_event(dataset, event_assig)
    recallsBCubed = functions.ev_functions_BCubed.evaluate_recall_event(dataset, event_assig)

    UTM31 = pyproj.Proj("+init=EPSG:32631")
    wgs84 = pyproj.Proj("+init=EPSG:4326")
    for k in xrange(K):
        for iclas, clas in enumerate(recalls):
            if clas[1]==k and clas[3]>0.1:
                print clas[0] + " Recall " + str(clas[3]) +"("+str(recallsBCubed[iclas][1])  + ") tweets: " +str(clas[2]) + " out of " + str(sum(dataset.tclass==clas[0]))
                latlong = pyproj.transform(UTM31, wgs84, *(mk_[k,:]*ln_std + ln_mean))
                stdlatlong = ((mk_[k,:]*ln_std + ln_mean)[0] + math.sqrt(np.linalg.inv(betamuk_[k]*Wk_[k,:,:])[0,0])*ln_std[0]
                ,(mk_[k,:]*ln_std + ln_mean)[1] + math.sqrt(np.linalg.inv(betamuk_[k] * Wk_[k, :, :])[1, 1])*ln_std[1])
                stdlatlong =  pyproj.transform(UTM31, wgs84, *stdlatlong)
                deltalong =  stdlatlong[0] - latlong[0]
                deltalat = stdlatlong[1] - latlong[1]
                print str(latlong[1])  + u" \u00B1 " + str(deltalat) + " " + str(latlong[0]) + u" \u00B1 " + str(deltalong)
                dev = math.sqrt(bk_[k] / ((ak_[k]-1) * betatauk_[k])) * tn_std
                print "Time: " + str(day + timedelta(0, mtauk_[k]*tn_std + tn_mean + 3600)) + u" \u00B1 " + str(timedelta(0, dev))
                print " --- "

    print "Top 10 words in each topic: "
    for t in xrange(T):
        print t, " ".join([vocabulary[i] for i in np.argsort(-phik_[t, :])[0:10]])

    fig = plt.figure(tight_layout=True)
    ax = fig.add_subplot(111, projection='3d')
    Knoise = np.argmax(np.bincount(event_assig.astype(int)))
    ax.scatter(ln[event_assig==Knoise,0], ln[event_assig==Knoise,1], tn[event_assig==Knoise], c='grey', marker='.', s=1, alpha = 0.25)
    ax.scatter(ln[event_assig!=Knoise,0], ln[event_assig!=Knoise,1], tn[event_assig!=Knoise], c = event_assig[event_assig!=Knoise])
    ax.set_title('Warble')
    ax.set_xlabel('lat',labelpad=0, size='small')
    ax.set_ylabel('long',labelpad=0, size='small')
    ax.set_zlabel('time',labelpad=-4, rotation=90, size='small')
    for axes in fig.axes:
        axes.xaxis.set_ticklabels([])
        axes.yaxis.set_ticklabels([])
        axes.zaxis.set_ticklabels([])
    fig.tight_layout()
    plt.show()
