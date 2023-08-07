import timeit
import numpy as np
import scipy as sp
from scipy.linalg import lapack

# References
# https://stackoverflow.com/questions/40703042/more-efficient-way-to-invert-a-matrix-knowing-it-is-symmetric-and-positive-semi
# https://stackoverflow.com/questions/44345340/efficiency-of-inverting-a-matrix-in-numpy-with-cholesky-decomposition
# https://stackoverflow.com/questions/39574924/why-is-inverting-a-positive-definite-matrix-via-cholesky-decomposition-slower-th

# Aux functions

inds_cache = {}
def upper_triangular_to_symmetric(ut):
    n = ut.shape[0]
    try:
        inds = inds_cache[n]
    except KeyError:
        inds = np.tri(n, k=-1, dtype="bool")
        inds_cache[n] = inds
    ut[inds] = ut.T[inds]

def inverse_accuracy(X, X_inv):
    return np.linalg.norm(X @ X_inv - I)

# Regular inverse

def full_inv1(X):
    return np.linalg.inv(X)

def full_inv2(X):
    return sp.linalg.inv(X)

# Cholesky based inverse

def chol_inv1(L):
    c = np.linalg.inv(L)
    return np.dot(c.T,c)

def chol_inv2(L):
    c = sp.linalg.inv(L)
    return np.dot(c.T,c)

def chol_inv3(L):
    inv, info = lapack.dpotri(L)
    if info != 0:
        raise ValueError('dpotri failed on input')
    upper_triangular_to_symmetric(inv)
    return inv

def chol_inv4(L,I):
    return sp.linalg.cho_solve((L, True), I)

def chol_inv5(L):
    I = np.eye(X.shape[0], dtype='int8')
    return sp.linalg.cho_solve((L, True), I, overwrite_b = True)

def pd_inv3(m):
    cholesky, info = lapack.dpotrf(m)
    if info != 0:
        raise ValueError('dpotrf failed on input {}'.format(m))
    inv, info = lapack.dpotri(cholesky)
    if info != 0:
        raise ValueError('dpotri failed on input {}'.format(cholesky))
    upper_triangular_to_symmetric(inv)
    return inv

def pd_inv4(m, chol):
    cholesky = chol(m)
    inv, info = lapack.dpotri(cholesky)
    if info != 0:
        raise ValueError('dpotri failed on input {}'.format(cholesky))
    upper_triangular_to_symmetric(inv)
    return inv


# Cholesky

def chol1(X):
    c = np.linalg.cholesky(X)
    return c

def chol2(X):
    c = sp.linalg.cholesky(X, lower=True)
    return c

def chol3(X):
    c, info = lapack.dpotrf(X, lower=True)
    if info != 0:
        raise ValueError('dpotrf failed on input {}'.format(m))
    return c

def chol4(X):
    c, _ = sp.linalg.cho_factor(X, lower=True)
    return c

# Condition number
def cond1(X):
    eigs = np.linalg.eigvalsh(X)
    return eigs[-1]/eigs[0]

def cond2(X):
    eigs = sp.linalg.eigvalsh(X)
    return eigs[-1]/eigs[0]

def cond3(X):
    small_e = sp.linalg.eigvalsh(X, subset_by_index=[0, 0])[0]
    large_e = sp.linalg.eigvalsh(X, subset_by_index=[X.shape[0] - 1, X.shape[0] - 1])[0]
    return large_e/small_e

def cond4(X):
    eigs = sp.linalg.eigh(X, eigvals_only=True)
    return eigs[-1]/eigs[0]

def cond5(X):
    small_e = sp.linalg.eigh(X, eigvals_only=True, subset_by_index=[0, 0])[0]
    large_e = sp.linalg.eigh(X, eigvals_only=True, subset_by_index=[X.shape[0] - 1, X.shape[0] - 1])[0]
    return large_e/small_e

def cond6(X):
    return np.linalg.cond(X)

# Generate matrix
N = 500
I = np.eye(N)
X = np.arange(N ** 2).reshape(N,N)
X = X + X.T - np.diag(X.diagonal()) + 0.1 * np.eye(N) #symmetry
X = np.dot(X,X.T) #positive-definite

TIMEIT_NUM=200

print("Inverse accuracy (norm(X@inv(X) - I)):")
print("full_inv1 norm = ", inverse_accuracy(X, full_inv1(X)))
print("full_inv2 norm = ", inverse_accuracy(X, full_inv2(X)))
print("chol_inv1(chol1) norm = ", inverse_accuracy(X, chol_inv1(chol1(X))))
print("chol_inv1(chol2) norm = ", inverse_accuracy(X, chol_inv1(chol2(X))))
print("chol_inv1(chol3) norm = ", inverse_accuracy(X, chol_inv1(chol3(X))))
print("chol_inv1(chol4) norm = ", inverse_accuracy(X, chol_inv1(chol4(X))))
print("chol_inv2(chol1) norm = ", inverse_accuracy(X, chol_inv2(chol1(X))))
print("chol_inv2(chol2) norm = ", inverse_accuracy(X, chol_inv2(chol2(X))))
print("chol_inv2(chol3) norm = ", inverse_accuracy(X, chol_inv2(chol3(X))))
print("chol_inv2(chol4) norm = ", inverse_accuracy(X, chol_inv2(chol4(X))))
print("chol_inv3(chol1) norm = ", inverse_accuracy(X, chol_inv3(chol1(X))))
print("chol_inv3(chol2) norm = ", inverse_accuracy(X, chol_inv3(chol2(X))))
print("chol_inv3(chol3) norm = ", inverse_accuracy(X, chol_inv3(chol3(X))))
print("chol_inv3(chol4) norm = ", inverse_accuracy(X, chol_inv3(chol4(X))))
print("chol_inv4(chol1) norm = ", inverse_accuracy(X, chol_inv4(chol1(X), I)))
print("chol_inv4(chol2) norm = ", inverse_accuracy(X, chol_inv4(chol2(X), I)))
print("chol_inv4(chol3) norm = ", inverse_accuracy(X, chol_inv4(chol3(X), I)))
print("chol_inv4(chol4) norm = ", inverse_accuracy(X, chol_inv4(chol4(X), I)))
print("chol_inv5(chol1) norm = ", inverse_accuracy(X, chol_inv5(chol1(X))))
print("chol_inv5(chol2) norm = ", inverse_accuracy(X, chol_inv5(chol2(X))))
print("chol_inv5(chol3) norm = ", inverse_accuracy(X, chol_inv5(chol3(X))))
print("chol_inv5(chol4) norm = ", inverse_accuracy(X, chol_inv5(chol4(X))))
print()

print("Condition number:")
print("cond1 norm = ", cond1(X))
print("cond2 norm = ", cond2(X))
print("cond3 norm = ", cond3(X))
print("cond4 norm = ", cond4(X))
print("cond5 norm = ", cond5(X))
print("cond6 norm = ", cond6(X))
print()

print("Cholesky timing:")
print("chol1 time = ", timeit.timeit('chol1(X)', globals=globals(), number=TIMEIT_NUM))
print("chol2 time = ", timeit.timeit('chol2(X)', globals=globals(), number=TIMEIT_NUM))
print("chol3 time = ", timeit.timeit('chol3(X)', globals=globals(), number=TIMEIT_NUM))
print("chol4 time = ", timeit.timeit('chol4(X)', globals=globals(), number=TIMEIT_NUM))
print()

L = chol1(X)

print("Inverse timing (Based on chol1):")
print("full_inv1 time = ", timeit.timeit('full_inv1(X)', globals=globals(), number=TIMEIT_NUM))
print("full_inv2 time = ", timeit.timeit('full_inv2(X)', globals=globals(), number=TIMEIT_NUM))
print("chol_inv1 time = ", timeit.timeit('chol_inv1(L)', globals=globals(), number=TIMEIT_NUM))
print("chol_inv2 time = ", timeit.timeit('chol_inv2(L)', globals=globals(), number=TIMEIT_NUM))
print("chol_inv3 time = ", timeit.timeit('chol_inv3(L)', globals=globals(), number=TIMEIT_NUM))
print("chol_inv4 time = ", timeit.timeit('chol_inv4(L, I)', globals=globals(), number=TIMEIT_NUM))
print("chol_inv5 time = ", timeit.timeit('chol_inv5(L)', globals=globals(), number=TIMEIT_NUM))
print()

print("Condition number:")
print("cond1 time = ", timeit.timeit('cond1(X)', globals=globals(), number=TIMEIT_NUM))
print("cond2 time = ", timeit.timeit('cond2(X)', globals=globals(), number=TIMEIT_NUM))
print("cond3 time = ", timeit.timeit('cond3(X)', globals=globals(), number=TIMEIT_NUM))
print("cond4 time = ", timeit.timeit('cond4(X)', globals=globals(), number=TIMEIT_NUM))
print("cond5 time = ", timeit.timeit('cond5(X)', globals=globals(), number=TIMEIT_NUM))
print("cond6 time = ", timeit.timeit('cond6(X)', globals=globals(), number=TIMEIT_NUM))