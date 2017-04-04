import numpy as np

def map_estimate(en_):
    return np.argmax(en_, axis = 1)

def precisionij(taggedj, clusti):
    return len(taggedj.ix[taggedj.tweet_id.isin(clusti.tweet_id)])*1./len(clusti)

def recallij(taggedi, clustj):
    return len(clustj.ix[clustj.tweet_id.isin(taggedi.tweet_id)])*1./len(taggedi)

def precision(clust, tagged):
    accum = 0
    for i in set(clust.cluster):
        clusti  = clust.ix[clust.cluster==i, :]
        precisions = [precisionij(tagged.ix[tagged.tclass == j,:], clusti) for j in set(tagged.tclass)]
        accum += len(clusti)*max(precisions)
    return accum*1./len(clust)

def recall(clust, tagged):
    accum = 0
    for i in set(tagged.tclass):
        taggedi  = tagged.ix[tagged.tclass==i, :]
        recalls = [recallij(taggedi,clust.ix[clust.cluster == j, :]) for j in set(clust.cluster)]
        accum += len(taggedi)*max(recalls)
    return accum*1./len(tagged)

def f_ij(taggedi, clustj):
    rec = recallij(taggedi,clustj)
    prec = precisionij(taggedi, clustj)
    if rec==0 and prec ==0:
        return 0
    else:
        return 2.*rec*prec/(rec+prec)

def f_measure(pr, re, alpha = 0.5):
    return 1/(alpha/pr + (1-alpha)/re)

def evaluate(dfcomp, en_est):
    dfcomp["cluster"] = en_est
    clust = dfcomp.ix[:,("tweet_id","cluster")]
    clas =  dfcomp.ix[:,("tweet_id","tclass")]
    purity = precision(clust,  clas)
    inv_purity = recall(clust, clas)
    f = f_measure(purity, inv_purity)
    return purity, inv_purity, f

def evaluate_wback(dfcomp, en_est):
    dfcomp["cluster"] = en_est
    Knoise = np.argmax(np.bincount(en_est.astype(int)))
    clust = dfcomp.ix[dfcomp["cluster"]!=Knoise,("tweet_id","cluster")]
    clas = dfcomp.ix[(dfcomp["tclass"]!="NonEvent"),("tweet_id","tclass")]
    purity = precision(clust,  clas)
    inv_purity = recall(clust, clas)
    f = f_measure(purity, inv_purity)
    return purity, inv_purity, f

def evaluate_recall_event(dfcomp,en_est):
    dfcomp["cluster"] = en_est
    Knoise = np.argmax(np.bincount(en_est.astype(int)))
    clust = dfcomp.ix[dfcomp["cluster"]!=Knoise,("tweet_id","cluster")]
    clas = dfcomp.ix[(dfcomp["tclass"]!="NonEvent"),("tweet_id","tclass")]
    accum = []
    for i in set(clas.tclass):
        taggedi  = clas.ix[clas.tclass==i, :]
        accum.append((i,max([recallij(taggedi,clust.ix[clust.cluster == j, :]) for j in set(clust.cluster)])))
    return accum