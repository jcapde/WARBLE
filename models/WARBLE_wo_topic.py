__author__ = 'jcapde87'
import math

from functions.inf_functions import *
from functions.init_functions import *


def lowerbound(values, bins, H, xedges, yedges, tn, ln, wnm, a_pi, pi_, en_,mk_, m_mu, betamuk_,beta_mu, nuk_, nu_Delta, Wk_, W_Delta,mtauk_, m_tau, betatauk_, beta_tau, ak_, a_lambda, bk_, b_lambda, thetak_, a_theta, znm_, Phi):
    E3 = E2 = H2 = E6 =  E9 = H9 = E10 =  0

    N, M = wnm.shape
    K, T = thetak_.shape

    if np.isscalar(a_pi):
        a_pi = np.array([a_pi]*K)

    arr_a_theta = np.array([a_theta]*T)

    E1 = -log_beta_function(a_pi) + np.dot((a_pi-np.ones(K)), dirichlet_expectation(pi_))
    H1 = log_beta_function(pi_) - np.dot((pi_-np.ones(K)), dirichlet_expectation(pi_))

    logdet = np.log(np.array([np.linalg.det(Wk_[k,:,:]) for k in xrange(K)]))
    logDeltak = psi(nuk_/2.) + psi((nuk_-1.)/2.) + 2.*np.log(2.) + logdet
    loglambdak = psi(ak_)-np.log(bk_)
    Elogtheta_ = dirichlet_expectation(thetak_)

    for n in range(N):
        wordids = np.where(wnm[n, :] != 0)[0]
        cnts = wnm[n, wordids]
        CNTS = np.tile(np.reshape(cnts,(len(wordids),1)),(1,T))
        E2 += np.dot(en_[n,:], dirichlet_expectation(pi_))
        H2 += -np.dot(en_[n,:], log_(en_[n,:]))
        product = np.array([np.dot(np.dot(ln[n,:]-mk_[k,:],Wk_[k,:,:]),(ln[n,:]-mk_[k,:]).T) for k in xrange(K)])
        E3 += 1./2*np.dot(en_[n,0:(K-1)],(logDeltak[0:(K-1)] -2.*np.log(2*math.pi) - nuk_[0:(K-1)]*product[0:(K-1)] - 2./betamuk_[0:(K-1)]).T)
        xbin = np.digitize(np.array([ln[n,0]]),xedges)[0]-2
        ybin = np.digitize(np.array([ln[n,1]]),yedges)[0]-2
        E3 += en_[n,(K-1)]*np.log(H[xbin,ybin]+ np.finfo(np.float32).eps)
        E6 += 1./2*np.dot(en_[n,0:(K-1)],loglambdak[0:(K-1)]-np.log(2.*math.pi)-1./betatauk_[0:(K-1)]-ak_[0:(K-1)]/bk_[0:(K-1)]*(tn[n]-mtauk_[0:(K-1)])**2)
        E6 += en_[n,(K-1)]*np.log(values[np.digitize(np.array([tn[n]]),bins)[0]-1])
        E9 += np.dot(en_[n, :], np.dot(znm_[n, wordids, :]*CNTS, Elogtheta_.T).sum(axis=0))
        H9 += -np.sum(znm_[n, wordids,:]*log_(znm_[n, wordids, :])*CNTS)
        E10 += np.sum(znm_[n, wordids,:]*log_(Phi[:, wordids]).T*CNTS)

    product = np.array([np.dot(np.dot(mk_[k,:]-m_mu,Wk_[k,:,:]), (mk_[k,:]-m_mu).T) for k in xrange(K)])
    traces = np.array([np.matrix.trace(np.dot(np.linalg.inv(W_Delta), Wk_[k,:,:])) for k in xrange(K)])

    H4 = np.sum((1. + np.log(2.*math.pi) - 1./2*(np.log(betamuk_) + logdet)))
    logB = nuk_/2.*logdet + nuk_*np.log(2.) + 1./2*np.log(math.pi) + gammaln(nuk_/2.) + gammaln((nuk_-1)/2.)
    H5 = np.sum((logB - (nuk_-3.)/2.*logDeltak + nuk_))
    E4 = np.sum((1./2*(np.log(beta_mu) + logDeltak - 2*np.log(2.*math.pi) - beta_mu*nuk_*product - 2.*beta_mu/betamuk_)))
    logB = nu_Delta/2.*np.log(np.linalg.det(W_Delta)) + nu_Delta*np.log(2.) + 1./2*np.log(math.pi) + gammaln(nu_Delta/2.) + gammaln((nu_Delta-1)/2.)
    E5 = np.sum((-logB + (nu_Delta-3.)/2.*logDeltak - nuk_/2.*traces))

    H7 = np.sum((1./2*(1. + np.log(2.*math.pi) - np.log(betatauk_*ak_/bk_)))[0:(K-1)])
    H8 = np.sum((gammaln(ak_) - (ak_-1.)*psi(ak_) - np.log(bk_) + ak_)[0:(K-1)])
    E7 = np.sum((1./2*(np.log(beta_tau)+loglambdak-np.log(2*math.pi)-beta_tau*ak_/bk_*(mtauk_-m_tau)**2-beta_tau/betatauk_))[0:(K-1)])
    E8 = np.sum((a_lambda*np.log(b_lambda) - gammaln(a_lambda) + (a_lambda-1.)*loglambdak - b_lambda*ak_/bk_)[0:(K-1)])
    E11 = np.sum([-log_beta_function(arr_a_theta) + np.dot((arr_a_theta-np.ones(T)), dirichlet_expectation(thetak_[k,:])) for k in xrange(K-1)])
    H11 = np.sum([log_beta_function(thetak_[k,:]) - np.dot((thetak_[k,:]-np.ones(T)),dirichlet_expectation(thetak_[k,:])) for k in xrange(K-1)])


    return E1 + E2 + E3 + E4 + E5 + E6 + E7 + E8 + E9 + E10 + E11 + H1 + H2 + H4 + H5 + H7 + H8 + H9 + H11


def inference_blei_wback(It, tn, ln, wnm, a_pi, T, K, m_mu, beta_mu, W_Delta, nu_Delta, m_tau, beta_tau, a_lambda, b_lambda, a_theta, Phi, values, bins, H, xedges, yedges):

    N, M = wnm.shape

    bk_ = np.zeros(K)
    Wk_ = np.zeros((K, 2, 2))
    Selk = np.zeros((K, 2, 2))

    en_ = init_kmeans(K, np.vstack((ln[:,0],ln[:,1],tn)).T)
    znm_ = init_z(wnm, N, M, T)

    Nek = np.sum(en_, axis=0)

    betatauk_ = beta_tau + Nek
    ak_ = a_lambda + Nek/2.
    mtauk_ = (m_tau*beta_tau + np.dot(en_.T, tn))/betatauk_

    betamuk_ = beta_mu + Nek
    nuk_ = nu_Delta + Nek
    lk_ = np.tile(1./Nek,(2,1)).T * np.dot(en_.T,ln)
    mk_ = np.tile(1./betamuk_,(2,1)).T * (m_mu*beta_mu + np.dot(en_.T,ln))

    pi_ = a_pi + Nek
    CNTS = np.tile(np.reshape(wnm, (N,M,1)), (1,1,T))
    thetak_ = a_theta + np.dot(en_.T, np.sum(znm_*CNTS, axis=1))

    for k in xrange(K):
        Selk[k, :, :] = 1./Nek[k]*np.dot((ln-lk_[k,:]).T,np.dot(np.diag(en_[:,k]),(ln-lk_[k,:])))
        aux = np.linalg.inv(W_Delta)+Nek[k]*Selk[k,:,:] + beta_mu*Nek[k]/(beta_mu + Nek[k])*np.dot(np.tile((lk_[k,:]-m_mu),(1,1)).T,np.tile(lk_[k,:]-m_mu,(1,1)))
        Wk_[k, :, :] = np.linalg.inv(aux)
        bk_[k] = b_lambda + 1./2*np.dot(en_[:,k].T,((tn-mtauk_[k])**2)) + 1./2*beta_tau*(mtauk_[k]-m_tau)**2

    it = 0
    lwbound_old = -1000.
    lwbound = lowerbound(values, bins, H, xedges, yedges, tn, ln, wnm, a_pi, pi_, en_,mk_, m_mu, betamuk_,beta_mu, nuk_, nu_Delta, Wk_, W_Delta,mtauk_, m_tau, betatauk_, beta_tau, ak_, a_lambda, bk_, b_lambda, thetak_, a_theta, znm_, Phi)
    print "Iteration: " + str(it) + " Lowerbound: " + str(lwbound) + " Diff: " + str(-(lwbound-lwbound_old)/lwbound)

    it +=1
    bounds = []
    bounds.append(lwbound)
    while it<It and abs((lwbound-lwbound_old)/lwbound)>1e-4:
        Elogpi_ = dirichlet_expectation(pi_)
        Elogtheta_ = dirichlet_expectation(thetak_)
        loglambdak = psi(ak_)-np.log(bk_)
        for n in xrange(N):
            wordids = np.where(wnm[n, :] != 0)[0]
            cnts = wnm[n, wordids]
            aux = np.exp(Elogpi_)
            for k in xrange(K-1):
                aux[k] *= np.exp(-np.log(2*math.pi) + 1./2*(2*np.log(2) + psi(nuk_[k]/2.)+psi((nuk_[k]-1.)/2.)+np.log(np.linalg.det(Wk_[k,:,:]))-2./betamuk_[k]-nuk_[k]*(np.dot((ln[n,:]-mk_[k,:]).T,np.dot(Wk_[k,:,:],(ln[n,:]-mk_[k,:]))))))
                aux[k] *= np.exp(1./2*loglambdak[k] - 1./2.*np.log(2*math.pi) - 1./(2.*betatauk_[k]) - ak_[k]/(2.*bk_[k])*(tn[n]-mtauk_[k])**2)
            # ST background
            xbin = np.digitize(np.array([ln[n,0]]),xedges)[0]-2
            ybin = np.digitize(np.array([ln[n,1]]),yedges)[0]-2
            aux[K-1] *= (H[xbin, ybin] + np.finfo(np.float32).eps)
            zbin = np.digitize(np.array([tn[n]]),bins)[0]-1
            aux[K-1] *= (values[zbin]+ np.finfo(np.float32).eps)
            # Textual
            aux *= np.exp(np.dot(np.reshape(cnts,(1, len(wordids))), np.dot(znm_[n, wordids, :],Elogtheta_.T)))[0]
            en_[n, :] = aux/np.sum(aux)
            aux2 = Phi[:, wordids].T*np.exp(np.tile(np.dot(en_[n, :], Elogtheta_), (len(wordids),1)))
            znm_[n, wordids, :] = np.copy(aux2/np.tile(aux2.sum(axis=1), (T,1)).T)

        Nek = np.sum(en_, axis=0)
        pi_ = a_pi + Nek
        CNTS = np.tile(np.reshape(wnm, (N,M,1)), (1,1,T))
        thetak_ = a_theta + np.dot(en_.T, np.sum(znm_*CNTS, axis=1))
        lk_ = np.tile(1./Nek,(2, 1)).T * np.dot(en_.T, ln)
        betamuk_ = beta_mu + Nek
        nuk_ = nu_Delta + Nek
        mk_ = np.tile(1./betamuk_,(2,1)).T * (m_mu*beta_mu + np.dot(en_.T,ln))
        betatauk_ = beta_tau + Nek
        ak_ = a_lambda + Nek/2.
        mtauk_ = (m_tau*beta_tau + np.dot(en_.T, tn))/betatauk_
        for k in xrange(K):
            Selk[k, :, :] = 1./Nek[k]*np.dot((ln-lk_[k,:]).T,np.dot(np.diag(en_[:,k]),(ln-lk_[k,:])))
            aux = np.linalg.inv(W_Delta)+Nek[k]*Selk[k,:,:] + beta_mu*Nek[k]/(beta_mu + Nek[k])*np.dot(np.tile((lk_[k,:]-m_mu),(1,1)).T,np.tile(lk_[k,:]-m_mu,(1,1)))
            Wk_[k, :, :] = np.linalg.inv(aux)
            bk_[k] = b_lambda + 1./2*np.dot(en_[:,k].T,((tn-mtauk_[k])**2)) + 1./2*beta_tau*(mtauk_[k]-m_tau)**2

        lwbound_old = np.copy(lwbound)
        lwbound = lowerbound(values, bins, H, xedges, yedges, tn, ln, wnm, a_pi, pi_, en_,mk_, m_mu, betamuk_,beta_mu, nuk_, nu_Delta, Wk_, W_Delta,mtauk_, m_tau, betatauk_, beta_tau, ak_, a_lambda, bk_, b_lambda, thetak_, a_theta, znm_, Phi)
        print "Iteration: " + str(it) + " Lowerbound: " + str(lwbound) + " Diff: " + str(-(lwbound-lwbound_old)/lwbound)
        it +=1
        bounds.append(lwbound)

    return en_, bounds

