import numpy as np
from numpy import linalg
from algos.GLASSO.base import base
from utils.common import np_soft_threshold
from utils.GLASSO.glasso import objective_F_cholesky

class OBN(base):
    def __init__(self, T,  N, lam, inner_T, ls_iter, step_lim):
        super(OBN, self).__init__(T, N, lam)
        self.inner_T = inner_T
        self.ls_iter = ls_iter
        self.step_lim = step_lim
        self.save_name = "OBN_N{N}_T{T}_innerT{inner_T}_LsIter{ls_iter}_StepLim{step_lim}" \
            .format(N=self.N, T=self.T, inner_T=self.inner_T, ls_iter=self.ls_iter, step_lim=self.step_lim)

    def compute(self, S, M, A0, status_f, history, test_check_f):
        As = []
        status = []

        lam = np.float32(self.lam)
        init_step = 1.0

        if A0 is None:
            A_diag = self.lam * np.ones(self.N, dtype='float32')
            A_diag = A_diag + np.diag(S)
            A_diag = 1.0 / A_diag
            A = np.diag(A_diag)
            A_diag = None
        else:
            A = np.array(A0, dtype='float32')
        A = M * A

        if history:
            As.append(A.copy())

        if status_f is not None: status.append(status_f(A, 0.0))

        for t in range(self.T):
            A_inv = np.linalg.inv(A)
            if test_check_f is not None:
                if test_check_f(A, S, self.lam, A_inv):
                    t -= 1
                    break

            if init_step != 0:
                sign_A = np.sign(A, dtype='float32')
                mask_A = np.abs(sign_A, dtype='float32').astype('int8')
                G = S - A_inv
                G_min = np_soft_threshold(G + lam*sign_A, lam*(1.0 - mask_A))
                sign_soft_G = np.sign(np_soft_threshold(G, lam), dtype='float32')
                mask = M
                Z = sign_A - np.bitwise_xor(mask, mask_A) * sign_soft_G
                sign_A = None
                sign_soft_G = None
                mask_A = None

                X = np.zeros(S.shape, dtype='float32')
                R = - mask * (G + lam*Z)
                G = None
                Q = R
                RR_old = np.sum(R * R, dtype='float32')
                epsilon = 1e-10
                inner_T = self.inner_T
                if inner_T < 0:
                    inner_T = 5 + t//(-inner_T)
                    if t % (-self.inner_T) == 0: init_step = 1.0
                for inner_t in range(np.minimum(S.shape[0],inner_T)):
                    if init_step == 0: break
                    if RR_old < epsilon: break
                    Y = mask * (A_inv @ Q @ A_inv)
                    alpha = RR_old / np.sum(Q * Y, dtype='float32')
                    X = X + alpha * Q
                    R = R - alpha * Y
                    RR = np.sum(R * R, dtype='float32')
                    beta = RR / RR_old
                    Q = R + beta * Q
                    RR_old = RR
                A_inv = None

                A, step = projected_linesearch_F(A, S, lam, G_min, Z, X, init=init_step, max_iter=self.ls_iter, step_lim=self.step_lim)
                if step == 0: init_step = 0

            if history:
                As.append(A.copy())

            if status_f is not None: status.append(status_f(A, step))

        if init_step == 0: t = np.inf
        return A, status, As, t+1

def projected_linesearch_F(A, S, lam, g_min, Z, Delta, init = 1, beta = 0.5, c = 0.1 , max_iter=10, step_lim=0):
    L = np.linalg.cholesky(A)
    init_F_val = objective_F_cholesky(A,S,lam,L)
    init_F_val_gA = init_F_val - c*np.sum(g_min*A, dtype='float32')
    init_F_val = None
    step = init
    for _ in range(max_iter):
        if step < step_lim: break
        try:
            A_next = A+step*Delta
            A_next[np.sign(A_next, dtype='float32') != Z] = 0
            A_next = A_next + np.transpose(A_next)
            A_next *= 0.5
            L_next = np.linalg.cholesky(A_next)
            if objective_F_cholesky(A_next,S,lam,L_next) <= init_F_val_gA + c*np.sum(g_min*A_next, dtype='float32'):
                    return A_next, step
        except linalg.LinAlgError:
            pass
        step *= beta
    return A, 0.0

def init_OBN_parser(OBN_pasrser):
    OBN_pasrser.set_defaults(algo='OBN')
    OBN_pasrser.add_argument(
        '-T', '--T', required=False, type=int, default=15, dest='T',
        help="Number of iterations.")
    OBN_pasrser.add_argument(
        '-inner_T', '--inner_T', required=False, type=int, default=50, dest='inner_T',
        help="Number of inner iterations.")
    OBN_pasrser.add_argument(
        '-linesearch', '--linesearch', required=False, type=int, default=15, dest='ls_iter',
        help="Number of linesearch iterations.")
    OBN_pasrser.add_argument(
        '-step_lim', '--step_limit', required=False, type=float, default=1e-4, dest='step_lim',
        help="The smallest step size possible.")
