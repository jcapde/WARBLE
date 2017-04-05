#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pickle
import argparse

from models.McInerneyBlei import inference_blei
from models.Tweet_SCAN import Tweet_SCAN
from models.WARBLE import inference_wback
from models.WARBLE_wo_topic import inference_blei_wback
from models.WARBLE_wo_background import inference_joint
from functions.ev_functions import *


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Learn topics from tweets')
    parser.add_argument('-T', metavar='T', type=int, default=30)
    parser.add_argument('-K', metavar='K', type=int, default=6)
    parser.add_argument('-maxIter', metavar='maxIter', type=int, default=50)
    args = parser.parse_args()


    ## LOAD DATA
    # Dataset
    dataset = pickle.load(open('data/tmp/dataset.pkl','rb'))
    # Word Matrix
    w = pickle.load(open('data/tmp/w.pkl', 'rb'))
    # Space time stats
    _, tn_std, _, ln_std = pickle.load(open('data/tmp/spacetime_stats.pkl', 'rb'))
    # User-Tweet relationship for Tweet-SCAN
    user_tweet = pickle.load(open('data/tmp/user_tweet.pkl', 'rb'))
    # Background
    xs, ys, HistLoc, ts, HistTemp = pickle.load(open('data/tmp/background.pkl','rb'))
    # Phi Matrix for non-joint topic estimation
    Phi = pickle.load(open('data/tmp/Phi.pkl', 'rb'))
    # Theta Matrix for Tweet-SCAn
    Theta = pickle.load(open('data/tmp/Theta.pkl', 'rb'))

    ## INITIALIZE VARIABLES

    N, M = w.shape
    T = args.T
    K = args.K
    maxIter = args.maxIter

    ## HYPERPARAMETERS
    # Proportions
    a_pi = np.array([0.1/K]*K)

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

    # Tweet-SCAN hyperparameters
    epsilon1 = 250./np.linalg.norm(ln_std)
    epsilon2 = 3600./tn_std
    epsilon3 = 0.9
    MinPts = 6

    ## Run models
    lwbound = [None]*4
    event_assig = np.zeros((5, N))
    print("Mcinerney & Blei model ----------------")
    en_, lwbound[0] = inference_blei(maxIter, tn, ln, w, a_pi, T, K, m_mu, beta_mu, W_Delta, nu_Delta, m_tau, beta_tau, a_lambda, b_lambda, a_theta, Phi)
    event_assig[0, :] = map_estimate(en_)
    print("WARBLE model w/o simulatenous topic learning model ----------------")
    en_, lwbound[1] = inference_blei_wback(maxIter, tn, ln, w, a_pi, T, K, m_mu, beta_mu, W_Delta, nu_Delta, m_tau, beta_tau, a_lambda, b_lambda, a_theta, Phi, HistTemp, ts, HistLoc, xs, ys)
    event_assig[1, :] = map_estimate(en_)
    print("WARBLE model w/o background model ----------------")
    en_, lwbound[2] = inference_joint(maxIter, tn, ln, w, a_pi, T, K, m_mu, beta_mu, W_Delta, nu_Delta, m_tau, beta_tau, a_lambda, b_lambda, a_theta, a_phi)
    event_assig[2, :] = map_estimate(en_)
    print("WARBLE model ----------------")
    en_, lwbound[3] = inference_wback(maxIter, tn, ln, w, a_pi, T, K, m_mu, beta_mu, W_Delta, nu_Delta, m_tau, beta_tau, a_lambda, b_lambda, a_theta, a_phi, HistTemp, ts, HistLoc, xs, ys)
    event_assig[3, :] = map_estimate(en_)
    print("Tweet-SCAN ----------------")
    clust = Tweet_SCAN()
    res_df = clust.run(ln, tn, Theta, user_tweet, epsilon1, epsilon2, epsilon3, MinPts, uparam=0.5)
    event_assig[4, :] = res_df["class"].values
    event_assig[4, np.where(event_assig[4, :] == -1)[0]] = (event_assig[4, :].max() + 1)

    ## Save results
    pickle.dump(event_assig, open('data/output/event_assignments.npy','wb'))
    pickle.dump(lwbound, open('data/output/lwbounds.pkl', 'wb'))

