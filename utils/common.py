import numpy as np

def np_soft_threshold(x, alphas):
    sign = np.sign(x, dtype='float32')
    return sign * np.maximum(np.abs(x, dtype='float32') - alphas, 0, dtype='float32')

def cp_soft_threshold(cp, x, alphas):
    sign = cp.sign(x, dtype='float32')
    return sign * cp.maximum(cp.abs(x, dtype='float32') - alphas, 0, dtype='float32')

def np_hard_threshold(x, alphas):
    x[np.abs(x, dtype='float32') <= alphas] = 0
    return x

def cp_hard_threshold(cp, x, alphas):
    x[cp.abs(x, dtype='float32') <= alphas] = 0
    return x