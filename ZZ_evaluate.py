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
    parser.add_argument('-K', metavar='K', type=int, default=7)
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
    MinPts = 7

    REP = 10
    purity_arr = np.zeros((REP,5))
    inv_purity_arr = np.zeros((REP,5))
    f_measure_arr = np.zeros((REP,5))
    event_assig = np.zeros((REP, 5, N))

    ## Run models

    for r in xrange(REP):
        print("Repetition: "+str(r)+"/"+str(REP))
        print("Mcinerney & Blei model ----------------")
        en_, _ = inference_blei(maxIter, tn, ln, w, a_pi, T, K, m_mu, beta_mu, W_Delta, nu_Delta, m_tau, beta_tau, a_lambda, b_lambda, a_theta, Phi)
        event_assig[r, 0, :] = map_estimate(en_)
        pr, re, f = evaluate(dataset, event_assig[r, 0,:])
        purity_arr[r, 0], inv_purity_arr[r, 0], f_measure_arr[r, 0] = pr, re, f

        print("WARBLE model w/o simulatenous topic learning model ----------------")
        en_, _ = inference_blei_wback(maxIter, tn, ln, w, a_pi, T, K, m_mu, beta_mu, W_Delta, nu_Delta, m_tau, beta_tau, a_lambda, b_lambda, a_theta, Phi, HistTemp, ts, HistLoc, xs, ys)
        event_assig[r, 1, :] = map_estimate(en_)
        pr, re, f = evaluate(dataset, event_assig[r, 1,:])
        purity_arr[r, 1], inv_purity_arr[r, 1], f_measure_arr[r, 1] = pr, re, f

        print("WARBLE model w/o background model ----------------")
        en_, _ = inference_joint(maxIter, tn, ln, w, a_pi, T, K, m_mu, beta_mu, W_Delta, nu_Delta, m_tau, beta_tau, a_lambda, b_lambda, a_theta, a_phi)
        event_assig[r, 2, :] = map_estimate(en_)
        pr, re, f = evaluate(dataset, event_assig[r, 2,:])
        purity_arr[r, 2], inv_purity_arr[r, 2], f_measure_arr[r, 2] = pr, re, f

        print("WARBLE model ----------------")
        en_, _ = inference_wback(maxIter, tn, ln, w, a_pi, T, K, m_mu, beta_mu, W_Delta, nu_Delta, m_tau, beta_tau, a_lambda, b_lambda, a_theta, a_phi, HistTemp, ts, HistLoc, xs, ys)
        event_assig[r, 3, :] = map_estimate(en_)
        pr, re, f = evaluate(dataset, event_assig[r, 3,:])
        purity_arr[r, 3], inv_purity_arr[r, 3], f_measure_arr[r, 3] = pr, re, f

        print("Tweet-SCAN ----------------")
        clust = Tweet_SCAN()
        res_df = clust.run(ln, tn, Theta, user_tweet, epsilon1, epsilon2, epsilon3, MinPts, uparam=0.5)
        event_assig[r, 4, :] = res_df["class"].values
        event_assig[r, 4, np.where(event_assig[r, 4, :] == -1)[0]] = (event_assig[r, 4, :].max() + 1)
        pr, re, f = evaluate(dataset, event_assig[r, 4,:])
        purity_arr[r, 4], inv_purity_arr[r, 4], f_measure_arr[r, 4] = pr, re, f

    ## Save results
    pickle.dump(event_assig, open('data/output/event_assignments_'+REP+'.npy','wb'))
    np.savetxt('data/output/purity_'+REP+'.txt', purity_arr)
    np.savetxt('data/output/inv_purity_'+REP+'.txt', inv_purity_arr)
    np.savetxt('data/output/f_measure_'+REP+'.txt', f_measure_arr)


