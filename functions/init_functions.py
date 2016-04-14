import numpy as np
from sklearn.cluster import KMeans

def init_z(wnm, N, M, T):
    znm_ = np.zeros((N, M, T))
    for n in xrange(N):
        wordids = np.where(wnm[n, :] != 0)[0]
        for m in wordids:
            aux = np.ones(T)
            znm_[n,m,:] = np.random.dirichlet(aux, 1)
    return znm_

def init_kmeans(K, xn):
    N = xn.shape[0]
    en_ = 0.1/(K-1)*np.ones((N, K))
    labels = KMeans(K).fit(xn).predict(xn)
    for i, lab in enumerate(labels):
        en_[i,lab] = 0.9
    return en_

