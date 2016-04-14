import numpy as np
from scipy.special import psi, gammaln

def log_(x):
    return np.log(x + np.finfo(np.float32).eps)

def log_beta_function(x):
    return np.sum(gammaln(x + np.finfo(np.float32).eps))-gammaln(np.sum(x + np.finfo(np.float32).eps))

def dirichlet_expectation(alpha):
    if len(alpha.shape) == 1:
        return psi(alpha + np.finfo(np.float32).eps) - psi(np.sum(alpha))
    return psi(alpha + np.finfo(np.float32).eps) - psi(np.sum(alpha, 1))[:, np.newaxis]
