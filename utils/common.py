import numpy as np
import scipy as sp

def np_soft_threshold(x, alphas):
    sign = np.sign(x, dtype='float32')
    return sign * np.maximum(np.abs(x, dtype='float32') - alphas, 0, dtype='float32')

def np_soft_threshold64(x, alphas):
    sign = np.sign(x, dtype='float64')
    return sign * np.maximum(np.abs(x, dtype='float64') - alphas, 0, dtype='float64')

def cp_soft_threshold(cp, x, alphas):
    sign = cp.sign(x, dtype='float32')
    return sign * cp.maximum(cp.abs(x, dtype='float32') - alphas, 0, dtype='float32')

def np_hard_threshold(x, alphas):
    x[np.abs(x, dtype='float32') <= alphas] = 0
    return x

def cp_hard_threshold(cp, x, alphas):
    x[cp.abs(x, dtype='float32') <= alphas] = 0
    return x

def np_cholesky_inv(x):
    I = np.eye(x.shape[0], dtype='int8')
    return sp.linalg.cho_solve((x, False), I, overwrite_b = True)

def np_cholesky(x):
    return sp.linalg.cho_factor(x, lower=False)[0]

def np_is_diag(M):
    m = M.shape[0]
    p,q = M.strides
    M_nodiag = np.lib.stride_tricks.as_strided(a[:,1:], (m-1,m), (p+q,q))
    return (M_nodiag == 0).all()