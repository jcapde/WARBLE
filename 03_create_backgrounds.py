#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pickle
import pyproj
import argparse

import numpy as np
import matplotlib.pyplot as plt

from  scipy.ndimage import gaussian_filter
from scipy.fftpack import rfft, rfftfreq, irfft

from datetime import datetime, timedelta
from pandas import read_pickle


def spatial_Gaussian_filtering(avg_loc):

    # Spatial Gaussian filter
    H, xedges, yedges = np.histogram2d(avg_loc[1:,0], avg_loc[1:,1], bins=(40, 40), normed = True)
    H = np.copy(H.T)
    im = plt.imshow(H, interpolation='nearest', origin='low', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
    plt.show()
    Hout = gaussian_filter(H, 1.5)
    H = np.copy(Hout)
    np.abs(xedges[:len(xedges)-1]-xedges[1:])[0]*np.abs(yedges[:len(yedges)-1]-yedges[1:])[0]*np.sum(H)
    im = plt.imshow(H, interpolation='nearest', origin='low', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
    plt.show()
    H = np.copy(H.T)

    return xedges, yedges, H


def temporal_FFT_IFFT(avg_time):

    # Temporal FFT-IFFT
    values, bins, patch = plt.hist(avg_time[1:], bins = 100, normed=True)
    plt.show()
    w = rfft(values)
    f = rfftfreq(len(values), bins[1]-bins[0])

    cutoff_idx = f > 0.6
    w = w.copy()
    w[cutoff_idx] = 0
    y = irfft(w)

    plt.plot(bins[0:100], y)
    plt.show()

    return bins, y

def filter_by_days(dataset, day, numerOfdays):

    th_l = day
    th_u = day + timedelta(days=numerOfdays)
    dataset = dataset.ix[(dataset.time>th_l) & (dataset.time<=th_u),:]
    dataset.index = range(len(dataset))

    return dataset.sort_values("time")

def compute_daily_space_time_histograms(dataset, day, ndays, tn_mean, tn_std, ln_mean, ln_std):

    UTM31 = pyproj.Proj("+init=EPSG:32631")
    wgs84 = pyproj.Proj("+init=EPSG:4326")

    avg_time = np.zeros(1)
    avg_loc = np.zeros((1,2))
    # Background models

    for i in xrange(ndays):
        th_l = day + timedelta(days=i)
        th_u = day + timedelta(days=i+1)

        selection = dataset.ix[(dataset.time > th_l) & (dataset.time <= th_u),:]
        selection.index = xrange(selection.shape[0])
        selection = selection.sort_values("time")

        selection.loc[:,"time"] = (selection["time"].values - min(selection["time"].values))

        aux = np.zeros(shape=(selection.shape[0], 2))
        for n in xrange(selection.shape[0]):
            aux[n, :] = pyproj.transform(wgs84, UTM31, *selection.ix[n, ["coordx", "coordy"]])
        aux_loc = np.copy((aux - ln_mean) / ln_std)

        aux_time = (np.array([delta.total_seconds() for delta in selection.time]) - tn_mean) / tn_std

        # need to update this with outlier removal process

        mask = np.ones(aux_loc.shape[0], dtype=bool)
        if len(mask) != 0:
            mask[np.argmin(aux_loc, axis=0)[0]] = False
            aux_loc = np.copy(aux_loc[mask, :])
            aux_time = np.copy(aux_time[mask])

        avg_time = np.append(avg_time, aux_time)
        avg_loc = np.append(avg_loc, aux_loc, axis=0)

    return avg_time, avg_loc


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Preprocess tweets')
    parser.add_argument('-fileName', metavar='fileName', type=str, default='Twitter-DS_MERCE_2014_tweets.pkl')
    parser.add_argument('-day', metavar='day', type=str, default='19/09/2014')
    parser.add_argument('-ndays', metavar='ndays', type=int, default=4)

    args = parser.parse_args()
    fileName = args.fileName
    day = datetime.strptime(args.day,'%d/%m/%Y')
    ndays = args.ndays

    tn_mean, tn_std, ln_mean, ln_std = pickle.load(open('data/tmp/spacetime_stats.pkl','rb'))

    fulldataset = read_pickle('data/input/' + fileName)
    avg_time, avg_loc = compute_daily_space_time_histograms(fulldataset, day, ndays, tn_mean, tn_std, ln_mean, ln_std)

    xs, ys, HistLoc = spatial_Gaussian_filtering(avg_loc)
    ts, HistTemp =  temporal_FFT_IFFT(avg_time)

    pickle.dump([xs, ys, HistLoc, ts, HistTemp], open('data/tmp/background.pkl','wb'))








