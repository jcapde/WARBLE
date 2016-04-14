#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pickle
from models.McInerneyBlei import inference_blei
from models.Tweet_SCAN import Tweet_SCAN
from models.WARBLE import inference_wback
from models.WARBLE_wo_topic import inference_blei_wback
from models.WARBLE_wo_background import inference_joint
from functions.ev_functions import *

if __name__ == "__main__":

    ln = np.load('data/input/ln.npy')
    tn = np.load('data/input/tn.npy')
    ln_std = np.load('data/input/ln_std.npy')
    tn_std = np.load('data/input/tn_std.npy')
    wnm = np.load('data/input/wnm.npy')
    Phi = np.load('data/input/Phi.npy')
    lab = np.load('data/input/label.npy')
    xedges = np.load('data/input/xedges.npy')
    yedges = np.load('data/input/yedges.npy')
    H = np.load('data/input/H.npy')
    bins = np.load('data/input/bins.npy')
    values = np.load('data/input/values.npy')
    dfcomp = pickle.load(open('data/input/dfcomp.pkl','rb'))
    C = np.load('data/input/C.npy')
    D = pickle.load(open('data/input/D.pkl','rb'))

    # INFERENCE
    # Maximum number of iterations
    It = 50


    # Model parameters
    N, M = wnm.shape
    T = 30
    K = 7

    # HYPERPARAMETERS
    # Proportions
    a_pi = np.array([0.1/K]*K)
    # Spatial features
    m_mu = np.mean(ln, axis=0)
    beta_mu = 9./(10*np.linalg.norm(ln.max(axis=0)-ln.min(axis=0))**2)
    W_Delta = 10*np.eye(2)
    nu_Delta = 100.
    # Temporal features
    m_tau = np.mean(tn)
    beta_tau = 9/(max(tn)-min(tn))**2
    a_lambda = 100.
    b_lambda = 1.
    # Topic proportions
    a_theta = 0.1
    # Word proportions
    a_phi = 0.1


    lwbound = [None]*4
    en_arr_est = np.zeros((5, N))
    en_arr = np.zeros((5, N, K))

    # Tweet-SCAN parameters
    epsilon1 = 250./np.linalg.norm(ln_std)
    epsilon2 = 3600./tn_std
    epsilon3 = 0.9
    MinPts = 7

    print "Blei&Mcinerney model -"
    en_arr[0, :, :], lwbound[0] = inference_blei(It, tn, ln, wnm, a_pi, T, K, m_mu, beta_mu, W_Delta, nu_Delta, m_tau, beta_tau, a_lambda, b_lambda, a_theta, Phi)
    print "WARBLE model without simulatenous topic learning model -"
    en_arr[1, :, :], lwbound[1] = inference_blei_wback(It, tn, ln, wnm, a_pi, T, K, m_mu, beta_mu, W_Delta, nu_Delta, m_tau, beta_tau, a_lambda, b_lambda, a_theta, Phi, values, bins, H, xedges, yedges)
    print "Joint topics-event inference Model"
    en_arr[2, :, :], lwbound[2] = inference_joint(It, tn, ln, wnm, a_pi, T, K, m_mu, beta_mu, W_Delta, nu_Delta, m_tau, beta_tau, a_lambda, b_lambda, a_theta, a_phi)
    print "Joint topics-event inference with Background Model"
    en_arr[3, :, :], lwbound[3] = inference_wback(It, tn, ln, wnm, a_pi, T, K, m_mu, beta_mu, W_Delta, nu_Delta, m_tau, beta_tau, a_lambda, b_lambda, a_theta, a_phi, values, bins, H, xedges, yedges)
    print "Tweet-SCAN"
    clust = Tweet_SCAN()
    en_arr[4, :, 0] = clust.run(ln, tn, C, D, epsilon1, epsilon2, epsilon3, MinPts, uparam=0.5)["class"].values

    np.save('data/output/en.npy', en_arr)
    pickle.dump([lwbound[0], lwbound[1], lwbound[2], lwbound[3]], open('data/output/lwbounds.pkl', 'wb'))

