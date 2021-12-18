import numpy as np
from numpy import linalg
from algos.GLASSO.base import base
from utils.common import np_soft_threshold
from utils.GLASSO.glasso import objective_f_cholesky

class NL_fista(base):
    def __init__(self, T,  N, lam, inner_T, ls_iter, step_lim):
        super(NL_fista, self).__init__(T, N, lam)
        self.inner_T = inner_T
        self.ls_iter = ls_iter
        self.step_lim = step_lim
        self.save_name = "NL_fista_N{N}_T{T}_innerT{inner_T}_LsIter{ls_iter}_StepLim{step_lim}" \
            .format(N=self.N, T=self.T, inner_T=self.inner_T, ls_iter=self.ls_iter, step_lim=self.step_lim)

    def compute(self, S, A0, status_f, history, test_check_f):
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

        if history:
            As.append(A.copy())

        if status_f is not None: status.append(status_f(A, 0.0))

        for t in range(self.T):
            A_inv = np.linalg.inv(A)
            if test_check_f is not None:
                if test_check_f(A, S, self.lam, A_inv):
                    break

            sign_A = np.sign(A, dtype='float32')
            mask_A = np.abs(sign_A, dtype='float32').astype('int8')
            G = S - A_inv
            F_subgrad_norm = np.linalg.norm(np_soft_threshold(G + lam*sign_A, lam*(1.0-mask_A)))
            sign_A = None
            mask_G = np.abs(np.sign(np_soft_threshold(G, lam), dtype='float32')).astype('int8')
            mask = np.bitwise_or(mask_A, mask_G)
            mask_G = None
            mask_A = None
            G_A_inv = G - A_inv

            inner_A = A
            inv_in_inv = A
            step = np.linalg.eigvalsh(A)[0] ** 2
            step = 1 / step
            t_k = 1
            for inner_t in range(self.inner_T):
                if init_step == 0: break
                a = G_A_inv + inv_in_inv
                inv_in_inv = None
                inner_A_next = mask * np_soft_threshold(inner_A - (1 / step) * a, lam / step)
                t_k_next = 0.5 * (1 + np.sqrt(1+4*t_k*t_k))
                inner_A = inner_A_next + (t_k - 1) / t_k_next * (inner_A_next - inner_A)
                t_k = t_k_next
                a = None
                inv_in_inv = A_inv@inner_A@A_inv
                sign_inner_A = np.sign(inner_A, dtype='float32')
                mask_inner_A = np.abs(sign_inner_A, dtype='float32').astype('int8')
                inner_f_subgrad_min = np.linalg.norm(np_soft_threshold(G + inv_in_inv + lam*sign_inner_A, lam*mask_inner_A))
                sign_inner_A = None
                mask_inner_A = None
                if inner_f_subgrad_min < 0.1 * F_subgrad_norm: break

            A, step = armijo_linesearch_F(A, S, lam, G, A_inv, inner_A - A, init=init_step, max_iter=self.ls_iter, step_lim=self.step_lim)
            if step == 0: init_step = 0

            if history:
                As.append(A.copy())

            if status_f is not None: status.append(status_f(A, step))

        if init_step == 0: t = np.inf
        return A, status, As, t+1

def objective_Q(A, A_next, G, A_inv):
    A_next_A = A_next - A
    return np.trace(A_next_A @ G, dtype='float32') +\
                0.5 * (np.sum(np.square(A_next_A@A_inv, dtype='float32'), dtype='float32'))

def armijo_linesearch_F(A, S, lam, g, A_inv, Delta, init = 1, beta = 0.5, c = 0.1 , max_iter=10, step_lim=0):
    L = np.linalg.cholesky(A)
    init_F_val = objective_f_cholesky(A,S,L)
    step = init
    for _ in range(max_iter):
        if step < step_lim: break
        try:
            A_next = A+step*Delta
            A_next = A_next + np.transpose(A_next)
            A_next *= 0.5
            L_next = np.linalg.cholesky(A_next)
            if objective_f_cholesky(A_next,S,L_next) <= init_F_val + step*c*objective_Q(A, A_next, g, A_inv):
                    return A_next, step
        except linalg.LinAlgError:
            pass
        step *= beta
    return A, 0.0

def init_NL_fista_parser(NL_fista_pasrser):
    NL_fista_pasrser.set_defaults(algo='NL_fista')
    NL_fista_pasrser.add_argument(
        '-T', '--T', required=False, type=int, default=15, dest='T',
        help="Number of iterations.")
    NL_fista_pasrser.add_argument(
        '-inner_T', '--inner_T', required=False, type=int, default=50, dest='inner_T',
        help="Number of inner iterations.")
    NL_fista_pasrser.add_argument(
        '-linesearch', '--linesearch', required=False, type=int, default=15, dest='ls_iter',
        help="Number of linesearch iterations.")
    NL_fista_pasrser.add_argument(
        '-step_lim', '--step_limit', required=False, type=float, default=1e-4, dest='step_lim',
        help="The smallest step size possible.")
