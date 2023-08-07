import numpy as np
from utils.common import np_soft_threshold
from utils.common import cp_soft_threshold

def objective_g(A_inv,S):
    return S-A_inv

def subgrad_min(A,S,lam):
    A_inv = np.linalg.inv(A)
    grad = objective_g(A_inv,S)+lam*np.sign(A)
    mask = 1.0 - np.abs(np.sign(A))
    return np_soft_threshold(grad, lam*mask)

def objective_f_cholesky(A,S,L):
    return -2*np.sum(np.log(np.diagonal(L),dtype='float32'))+np.trace(S@A ,dtype='float32')

def objective_F_cholesky(A,S,lam,L):
    return objective_f_cholesky(A,S,L)+lam*np.sum(np.abs(A))

def objective_f_cholesky64(A,S,L):
    return -2*np.sum(np.log(np.diagonal(L),dtype='float64'))+np.trace(S@A ,dtype='float64')

def objective_F_cholesky64(A,S,lam,L):
    return objective_f_cholesky64(A,S,L)+lam*np.sum(np.abs(A))

def objective_dual(A_inv,S,lam):
    U = np.minimum(np.maximum(A_inv - S, -lam), lam)
    n = S.shape[0]
    det_sign, logdet_val = np.linalg.slogdet(U+S)
    if det_sign <= 0: return -np.inf
    return -logdet_val-n

def cuda_subgrad_min(cp,A,S,lam):
    A_inv = cp.linalg.inv(A)
    grad = objective_g(A_inv,S)+lam*cp.sign(A)
    mask = 1.0 - cp.abs(cp.sign(A))
    return cp_soft_threshold(cp, grad, lam*mask)

def cuda_objective_f_cholesky(cp,A,S,L):
    return -2*cp.sum(cp.log(cp.diagonal(L),dtype='float32'))+cp.trace(S@A ,dtype='float32')

def cuda_objective_F_cholesky(cp, A,S,lam,L):
    return cuda_objective_f_cholesky(cp,A,S,L)+lam*cp.sum(cp.abs(A))

def cuda_objective_dual(cp, A_inv,S,lam):
    U = cp.minimum(cp.maximum(A_inv - S, -lam), lam)
    n = S.shape[0]
    det_sign, logdet_val = cp.linalg.slogdet(U+S)
    if det_sign <= 0: return -cp.inf
    return -logdet_val-n

def create_glasso_status(S, lam, true_A = None):
    def glasso_status(A, step):
        L = np.linalg.cholesky(A)
        loss = objective_F_cholesky(A, S, lam, L)
        gap = loss + objective_dual(np.linalg.inv(A), S, lam)
        nnz = np.count_nonzero(A)
        grad = np.linalg.norm(subgrad_min(A, S, lam)) ** 2
        cond = np.linalg.cond(A)
        if true_A is not None:
            nmse = np.linalg.norm(A - true_A) / np.linalg.norm(true_A)
            nmse = nmse ** 2
        else:
            nmse = -1
        return gap, loss, nnz, grad, step, cond, nmse

    return glasso_status

def cuda_SPD_cond(cp, A):
    eig = cp.abs(cp.linalg.eigvalsh(A))
    return cp.max(eig) / cp.min(eig)

def cuda_create_glasso_status(cp, S, lam, true_A = None):
    S = cp.array(S)
    def glasso_status(A, step):
        L = cp.linalg.cholesky(A)
        loss = cuda_objective_F_cholesky(cp, A, S, lam, L)
        gap = loss + cuda_objective_dual(cp, cp.linalg.inv(A), S, lam)
        nnz = cp.count_nonzero(A)
        grad = cp.linalg.norm(cuda_subgrad_min(cp, A, S, lam)) ** 2
        cond = cuda_SPD_cond(cp, A)
        if true_A is not None:
            nmse = cp.linalg.norm(A - true_A) / cp.linalg.norm(true_A)
            nmse = nmse ** 2
        else:
            nmse = -1
        return gap, loss, nnz, grad, step, cond, nmse

    return glasso_status

def gap_test_check(A, S, lam, epsilon, A_inv = None):
    L = np.linalg.cholesky(A)
    loss = objective_F_cholesky(A, S, lam, L)
    L = None
    if A_inv is None: A_inv = np.linalg.inv(A)
    gap = np.abs(loss + objective_dual(A_inv, S, lam))
    return gap < epsilon

def cuda_gap_test_check(cp, A, S, lam, epsilon, A_inv = None):
    L = cp.linalg.cholesky(A)
    loss = cuda_objective_F_cholesky(cp, A, S, lam, L)
    L = None
    if A_inv is None: A_inv = cp.linalg.inv(A)
    gap = cp.abs(loss + cuda_objective_dual(cp, A_inv, S, lam))
    return gap < epsilon

def gap_rel_test_check(A, S, lam, epsilon, A_inv = None):
    L = np.linalg.cholesky(A)
    loss = objective_F_cholesky(A, S, lam, L)
    L = None
    if A_inv is None: A_inv = np.linalg.inv(A)
    gap_rel = np.abs(loss / objective_dual(A_inv, S, lam))
    return (1 >= gap_rel > 1 - epsilon) or (1 >= 1 / gap_rel > 1 - epsilon)

def cuda_gap_rel_test_check(cp, A, S, lam, epsilon, A_inv = None):
    L = cp.linalg.cholesky(A)
    loss = cuda_objective_F_cholesky(cp, A, S, lam, L)
    L = None
    if A_inv is None: A_inv = cp.linalg.inv(A)
    gap_rel = cp.abs(loss / cuda_objective_dual(cp, A_inv, S, lam))
    return (1 >= gap_rel > 1 - epsilon) or (1 >= 1 / gap_rel > 1 - epsilon)

def rel_test_check(A, S, lam, epsilon, A_inv = None):
    A_abs = np.abs(A)
    A_abs_sum = np.sum(A_abs)
    grad = objective_g(A_inv,S)+lam*np.sign(A)
    mask = 1.0 - np.sign(A_abs)
    A_abs = None
    subgrad_min = np_soft_threshold(grad, lam*mask)
    subgrad_min_abs_sum = np.sum(np.abs(subgrad_min))
    subgrad_min = None
    grad = None
    mask = None
    return subgrad_min_abs_sum/A_abs_sum < epsilon

def cuda_rel_test_check(cp, A, S, lam, epsilon, A_inv = None):
    A_abs = cp.abs(A)
    A_abs_sum = cp.sum(A_abs)
    grad = objective_g(A_inv,S)+lam*cp.sign(A)
    mask = 1.0 - cp.sign(A_abs)
    A_abs = None
    subgrad_min = cp_soft_threshold(cp, grad, lam*mask)
    subgrad_min_abs_sum = cp.sum(cp.abs(subgrad_min))
    subgrad_min = None
    grad = None
    mask = None
    return subgrad_min_abs_sum/A_abs_sum < epsilon

def nmse_test_check(A, epsilon, true_A):
    nmse = np.linalg.norm(A - true_A) / np.linalg.norm(true_A)
    nmse = nmse ** 2
    return nmse <= epsilon

def cuda_nmse_test_check(cp, A, epsilon, true_A):
    nmse = cp.linalg.norm(A - true_A) / cp.linalg.norm(true_A)
    nmse = nmse ** 2
    return nmse <= epsilon

def diff_test_check(A, S, lam, epsilon, min_loss):
    L = np.linalg.cholesky(A)
    loss = objective_F_cholesky(A, S, lam, L)
    return np.abs(loss-min_loss) <= epsilon

def cuda_diff_test_check(cp, A, S, lam, epsilon, min_loss):
    L = cp.linalg.cholesky(A)
    loss = cuda_objective_F_cholesky(cp, A, S, lam, L)
    return cp.abs(loss-min_loss) <= epsilon

def diff_rel_test_check(A, S, lam, epsilon, min_loss):
    L = np.linalg.cholesky(A)
    loss = objective_F_cholesky(A, S, lam, L)
    return np.abs(loss-min_loss) / min_loss <= epsilon

def cuda_diff_rel_test_check(cp, A, S, lam, epsilon, min_loss):
    L = cp.linalg.cholesky(A)
    loss = cuda_objective_F_cholesky(cp, A, S, lam, L)
    return cp.abs(loss-min_loss) / min_loss <= epsilon