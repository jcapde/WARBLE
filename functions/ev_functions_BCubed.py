import numpy as np
import multiprocessing
from functools import partial

def map_estimate(en_):
    return np.argmax(en_, axis = 1)

def correctness(e, e_):
    if (e.tclass == e_.tclass) and (e.cluster == e_.cluster):
        return 1.
    else:
        return 0.

def Precision_BCubed_wback(dfcomp):
    accum_clust = 0
    for e in dfcomp.index:
        c = dfcomp.ix[e,"cluster"]
        accum = 0
        clust_len = 0
        for e_ in dfcomp.ix[dfcomp["cluster"] == c,:].index:
            if dfcomp.ix[e, "tclass"] != "-- Background" or dfcomp.ix[e_, "tclass"] != "-- Background":
                accum += correctness(dfcomp.ix[e,:], dfcomp.ix[e_,:])
                clust_len +=1
        if clust_len != 0:
            accum_clust += accum*1./clust_len
        #print accum, clust_len, len(dfcomp.ix[dfcomp["cluster"] == c,:].index)
    return accum_clust*1./dfcomp.shape[0]

def Recall_BCubed_wback(dfcomp, Knoise):
    accum_class = 0
    for e in dfcomp.index:
        c = dfcomp.ix[e,"tclass"]
        accum = 0
        class_len = 0
        for e_ in dfcomp.ix[dfcomp["tclass"] == c,:].index:
            if dfcomp.ix[e, "cluster"] != Knoise or dfcomp.ix[e_, "cluster"] != Knoise:
                accum += correctness(dfcomp.ix[e,:], dfcomp.ix[e_,:])
                class_len +=1
        if class_len != 0:
            accum_class += accum/class_len
        #print accum, class_len, len(dfcomp.ix[dfcomp["tclass"] == c,:].index)
    return accum_class/dfcomp.shape[0]

def Precision_BCubed(dfcomp):
    accum_clust = 0
    for e in dfcomp.index:
        c = dfcomp.ix[e,"cluster"]
        accum = 0
        clust_len = len(dfcomp.ix[dfcomp["cluster"] == c,:].index)
        for e_ in dfcomp.ix[dfcomp["cluster"] == c,:].index:
            accum += correctness(dfcomp.ix[e,:], dfcomp.ix[e_,:])
        accum_clust += accum*1./clust_len
    return accum_clust*1./dfcomp.shape[0]

def Recall_BCubed(dfcomp):
    accum_class = 0
    for e in dfcomp.index:
        c = dfcomp.ix[e,"tclass"]
        accum = 0
        class_len = len(dfcomp.ix[dfcomp["tclass"] == c,:].index)
        for e_ in dfcomp.ix[dfcomp["tclass"] == c,:].index:
            accum += correctness(dfcomp.ix[e,:], dfcomp.ix[e_,:])
        accum_class += accum/class_len
    return accum_class/dfcomp.shape[0]

def PrecisionRecall_BCubed(dfcomp):
    accum_clust = 0
    accum_class = 0
    for e in dfcomp.index:
        cr = dfcomp.ix[e,"cluster"]
        cs = dfcomp.ix[e,"tclass"]
        accum = 0
        clust_len = len(dfcomp.ix[dfcomp["cluster"] == cr,:].index)
        for e_ in dfcomp.ix[dfcomp["cluster"] == cr,:].index:
            accum += correctness(dfcomp.ix[e,:], dfcomp.ix[e_,:])
        accum_clust += accum*1./clust_len
        accum = 0
        class_len = len(dfcomp.ix[dfcomp["tclass"] == cs,:].index)
        for e_ in dfcomp.ix[dfcomp["tclass"] == cs,:].index:
            accum += correctness(dfcomp.ix[e,:], dfcomp.ix[e_,:])
        accum_class += accum/class_len
    return accum_clust*1./dfcomp.shape[0], accum_class/dfcomp.shape[0]

def PR_par(e, dfcomp):
    cr = dfcomp.ix[e,"cluster"]
    cs = dfcomp.ix[e,"tclass"]
    accum = 0
    clust_len = len(dfcomp.ix[dfcomp["cluster"] == cr,:].index)
    for e_ in dfcomp.ix[dfcomp["cluster"] == cr,:].index:
        accum += correctness(dfcomp.ix[e,:], dfcomp.ix[e_,:])
    accum_clust = accum*1./clust_len
    accum = 0
    class_len = len(dfcomp.ix[dfcomp["tclass"] == cs,:].index)
    for e_ in dfcomp.ix[dfcomp["tclass"] == cs,:].index:
        accum += correctness(dfcomp.ix[e,:], dfcomp.ix[e_,:])
    accum_class = accum/class_len
    return accum_clust, accum_class


def PrecisionRecall_BCubed_Parallel(dfcomp):
    N = dfcomp.shape[0]
    pool = multiprocessing.Pool(2)
    aux = pool.map(partial(PR_par,dfcomp=dfcomp),dfcomp.index)
    return np.sum([el[0] for el in aux])/N, np.sum([el[1] for el in aux])/N


def F_BCubed(pr,re, alpha = 0.5):
    return 1/(alpha/pr + (1-alpha)/re)


def evaluate(dfcomp, en_est):
    dfcomp["cluster"] = en_est
    pr, re = PrecisionRecall_BCubed_Parallel(dfcomp)
    f = F_BCubed(pr, re)
    return pr, re, f


def evaluate_wback(dfcomp, en_est):
    dfcomp["cluster"] = en_est
    Knoise = np.argmax(np.bincount(dfcomp["cluster"].astype(int)))
    pr = Precision_BCubed_wback(dfcomp.ix[(dfcomp["cluster"]!=Knoise),:])
    re = Recall_BCubed_wback(dfcomp.ix[(dfcomp["tclass"]!="-- Background"),:], Knoise)
    f = F_BCubed(pr, re)
    return pr, re, f

def evaluate_recall_event(dfcomp,en_est):
    dfcomp["cluster"] = en_est
    Knoise = np.argmax(np.bincount(en_est.astype(int)))
    dfcomp = dfcomp.ix[(dfcomp["tclass"]!="-- Background"),:]
    aux = []

    for c in set(dfcomp.tclass):
        class_len = 0
        accum = 0
        for e in dfcomp.ix[dfcomp.tclass == c, :].index:
            for e_ in dfcomp.ix[dfcomp.tclass == c, :].index:
                if dfcomp.ix[e, "cluster"] != Knoise or dfcomp.ix[e_, "cluster"] != Knoise:
                    accum += correctness(dfcomp.ix[e,:], dfcomp.ix[e_,:])
                    class_len +=1.
        if class_len != 0:
            aux.append((c,accum/class_len))
        else:
            aux.append((c,0.))
    return aux

