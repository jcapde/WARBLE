#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pickle
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_pdf import PdfPages
from pandas import read_pickle


if __name__ == "__main__":
    REP = 10

    dataset = read_pickle('data/tmp/dataset.pkl')
    ln = dataset.as_matrix(["x", "y"])
    tn = dataset.t.values

    lab = np.zeros(dataset.shape[0])
    for i, c in enumerate(list(set(dataset.tclass))):
        lab[np.where(dataset.tclass == c)[0]] = i

    #type = 'BCubed_'
    type = ''
    en_arr = np.load('data/output/event_assignments_' + str(REP)+'.npy')
    purity_arr = np.loadtxt('data/output/purity_' + str(REP)+'.txt')
    inv_purity_arr = np.loadtxt('data/output/inv_purity_' + str(REP)+'.txt')
    f_measure_arr = np.loadtxt('data/output/f_measure_' + str(REP)+'.txt')

    print f_measure_arr


    bar_width = 0.25
    np.arange(5)
    plt.bar(np.arange(5), f_measure_arr.mean(axis=0), bar_width,
                     alpha=0.5,
                     color='r',
                     hatch="/",
                     yerr=f_measure_arr.std(axis=0),
                     label='F-measure')

    plt.bar(np.arange(5)+bar_width, purity_arr.mean(axis=0), bar_width,
                     alpha=0.25,
                     color='b',
                     hatch="\\",
                     yerr=purity_arr.std(axis=0),
                     label='Purity')
    plt.bar(np.arange(5)+2*bar_width, inv_purity_arr.mean(axis=0), bar_width,
                     alpha=0.25,
                     color='g',
                     hatch="x",
                     yerr=inv_purity_arr.std(axis=0),
                     label='Inverse Purity')

    #plt.xlabel('Event Model')
    #plt.ylabel('Scores')
    plt.xticks(np.arange(5) + bar_width, ('A', 'B', 'C', 'D', 'E'))
    plt.legend()
    plt.tight_layout()
    plt.show()

    #pp = PdfPages('./Fig_'+type+'Performance.pdf')
    #pp.savefig()
    #pp.close()

    rep = 2
    fig = plt.figure(tight_layout=True, figsize=(20,4))
    en_est = en_arr[rep, 0, :]
    ax = fig.add_subplot(161, projection='3d')
    ax.scatter(ln[:,0], ln[:,1], tn, c = en_est.astype(int))
    ax.set_title('A')
    ax.set_xlabel('lat',labelpad=0, size='small')
    ax.set_ylabel('long',labelpad=0, size='small')
    ax.set_zlabel('time',labelpad=-4, rotation=90, size='small')
    en_est = en_arr[rep, 1, :]
    ax = fig.add_subplot(162, projection='3d')
    Knoise = np.argmax(np.bincount(en_est.astype(int)))
    ax.scatter(ln[en_est==Knoise,0], ln[en_est==Knoise,1], tn[en_est==Knoise], c='grey', marker='.', s=1, alpha = 0.25)
    ax.scatter(ln[en_est!=Knoise,0], ln[en_est!=Knoise,1], tn[en_est!=Knoise], c = en_est[en_est!=Knoise])
    ax.set_title('B')
    ax.set_xlabel('lat',labelpad=0, size='small')
    ax.set_ylabel('long',labelpad=0, size='small')
    ax.set_zlabel('time',labelpad=-4, rotation=90, size='small')
    en_est = en_arr[rep, 2, :]
    ax = fig.add_subplot(163, projection='3d')
    Knoise = np.argmax(np.bincount(en_est.astype(int)))
    ax.scatter(ln[en_est==Knoise,0], ln[en_est==Knoise,1], tn[en_est==Knoise], c='grey', marker='.', s=1, alpha = 0.25)
    ax.scatter(ln[en_est!=Knoise,0], ln[en_est!=Knoise,1], tn[en_est!=Knoise], c = en_est[en_est!=Knoise])
    ax.set_title('C')
    ax.set_xlabel('lat',labelpad=0, size='small')
    ax.set_ylabel('long',labelpad=0, size='small')
    ax.set_zlabel('time',labelpad=-4, rotation=90, size='small')
    en_est = en_arr[rep, 3, :]
    ax = fig.add_subplot(164, projection='3d')
    Knoise = np.argmax(np.bincount(en_est.astype(int)))
    ax.scatter(ln[en_est==Knoise,0], ln[en_est==Knoise,1], tn[en_est==Knoise], c='grey', marker='.', s=1, alpha = 0.25)
    ax.scatter(ln[en_est!=Knoise,0], ln[en_est!=Knoise,1], tn[en_est!=Knoise], c = en_est[en_est!=Knoise])
    ax.set_title('D')
    ax.set_xlabel('lat',labelpad=0, size='small')
    ax.set_ylabel('long',labelpad=0, size='small')
    ax.set_zlabel('time',labelpad=-4, rotation=90, size='small')
    en_est = en_arr[rep, 4, :]
    ax = fig.add_subplot(165, projection='3d')
    Knoise = np.argmax(np.bincount(en_est.astype(int)))
    ax.scatter(ln[en_est==Knoise,0], ln[en_est==Knoise,1], tn[en_est==Knoise], c='grey', marker='.', s=1, alpha = 0.25)
    ax.scatter(ln[en_est!=Knoise,0], ln[en_est!=Knoise,1], tn[en_est!=Knoise], c = en_est[en_est!=Knoise])
    ax.set_title('E')
    ax.set_xlabel('lat',labelpad=0, size='small')
    ax.set_ylabel('long',labelpad=0, size='small')
    ax.set_zlabel('time',labelpad=-4, rotation=90, size='small')
    en_est = lab
    ax = fig.add_subplot(166, projection='3d')
    Knoise = np.argmax(np.bincount(en_est.astype(int)))
    ax.scatter(ln[en_est==Knoise,0], ln[en_est==Knoise,1], tn[en_est==Knoise], c='grey', marker='.', s=1, alpha = 0.25)
    ax.scatter(ln[en_est!=Knoise,0], ln[en_est!=Knoise,1], tn[en_est!=Knoise], c = en_est[en_est!=Knoise])
    ax.set_title('F')
    ax.set_xlabel('lat',labelpad=0, size='small')
    ax.set_ylabel('long',labelpad=0, size='small')
    ax.set_zlabel('time',labelpad=-4, rotation=90, size='small')

    for axes in fig.axes:
        axes.xaxis.set_ticklabels([])
        axes.yaxis.set_ticklabels([])
        axes.zaxis.set_ticklabels([])
    fig.tight_layout()
    plt.show()
    #pp = PdfPages('./Fig_Events.pdf')
    #pp.savefig()
    #pp.close()