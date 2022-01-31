import numpy as np
from numpy import linalg
from algos.GLASSO.base import base
from utils.common import np_soft_threshold
from utils.GLASSO.glasso import objective_F_cholesky
from algos.GLASSO.OBN import projected_linesearch_F

class pISTA_OBN(base):
    def __init__(self, T, N, lam, inner_T, ls_iter, step_lim, init_step):
        super(pISTA_OBN, self).__init__(T, N, lam)
        self.inner_T = inner_T
        self.ls_iter = ls_iter
        self.step_lim = step_lim
        self.init_step = init_step
        self.save_name = "pISTA_OBN_N{N}_T{T}_innerT{inner_T}_step{step}_LsIter{ls_iter}_StepLim{step_lim}"\
            .format(N=self.N, T=self.T, inner_T=self.inner_T, step=self.init_step, ls_iter=self.ls_iter, step_lim=self.step_lim)

    def compute(self, S, A0, status_f, history, test_check_f):
        init_step = np.float32(self.init_step)
        As = []
        status = []

        lam = np.float32(self.lam)

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
                    t -= 1
                    break

            if t % 2 == 0:
                sign_A = np.sign(A, dtype='float32')
                mask_A = np.abs(sign_A, dtype='float32').astype('int8')
                G = S - A_inv
                sign_soft_G = np.sign(np_soft_threshold(G, lam), dtype='float32')
                mask_G = np.abs(sign_soft_G).astype('int8')
                mask = np.bitwise_or(mask_A, mask_G)
                mask_G = None

                AgA = A @ (mask * G) @ A
                G = None

                sign_A -= np.bitwise_xor(mask, mask_A) * sign_soft_G
                sign_soft_G = None

                AhA = A @ sign_A @ A
                AhA *= lam
                A_diag = np.diag(A).copy().reshape(-1, 1)
                np.fill_diagonal(A, 0)
                A_no_diag = A
                AAt = ((A_no_diag * A_no_diag) + (A_diag * A_diag.T)) * sign_A
                AAt *= lam
                np.fill_diagonal(A, A_diag)
                A_diag = None
                A_no_diag = None
                sign_A = None

                AghA = AgA + AhA - AAt
                AgA = None
                AhA = None

                A, step = pista_cholesky_linesearch(A, S, lam, mask, AghA, AAt, A,
                                                            step=init_step, max_iter=self.ls_iter, step_lim=self.step_lim)
                #if step == 0: init_step = 0
            else:
                sign_A = np.sign(A, dtype='float32')
                mask_A = np.abs(sign_A, dtype='float32').astype('int8')
                G = S - A_inv
                G_min = np_soft_threshold(G + lam*sign_A, lam*(1.0 - mask_A))
                sign_soft_G = np.sign(np_soft_threshold(G, lam), dtype='float32')
                mask_G = np.abs(sign_soft_G).astype('int8')
                mask = np.bitwise_or(mask_A, mask_G)
                Z = sign_A - np.bitwise_xor(mask, mask_A) * sign_soft_G
                sign_A = None
                sign_soft_G = None
                mask_G = None
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

            if history:
                As.append(A.copy())

            if status_f is not None: status.append(status_f(A, step))

        if init_step == 0: t = np.inf
        return A, status, As, t+1

def pista_cholesky_linesearch(A, S, lam, mask, a, b, c, step, max_iter, step_lim):
    if step == 0:
        return A, 0.0
    beta = step
    L = np.linalg.cholesky(A)
    init_F_value = objective_F_cholesky(A,S,lam,L)
    L = None
    beta_psd = None
    for _ in range(max_iter):
        if beta < step_lim: break
        try:
            beta_a = beta * a
            beta_b = np.abs(beta * b, dtype='float32')
            A_next = mask * np_soft_threshold(c - beta_a, beta_b)
            beta_a = None
            beta_b = None

            A_next = A_next + np.transpose(A_next)
            A_next *= 0.5
            L = np.linalg.cholesky(A_next)
            if beta_psd is None: beta_psd = beta
            if objective_F_cholesky(A_next,S,lam,L) < init_F_value:
                return A_next, beta
        except linalg.LinAlgError:
            pass
        beta *= 0.5

    eigs = np.linalg.eigvalsh(A)
    beta = (eigs[0]/eigs[-1]) ** 2
    beta = np.float32(0.81 * beta)
    eigs = None
    if beta_psd is not None and beta > beta_psd: beta = beta_psd
    beta_eps = np.finfo(np.float32).eps
    while True:
        try:
            beta_a = beta * a
            beta_b = np.abs(beta * b, dtype='float32')
            A_next = mask * np_soft_threshold(c - beta_a, beta_b)
            beta_a = None
            beta_b = None

            A_next = A_next + np.transpose(A_next)
            A_next *= 0.5
            L = np.linalg.cholesky(A_next)
            return A_next, beta
        except linalg.LinAlgError:
            pass
        beta *= 0.5
        #Emulate do while
        if beta < beta_eps: break

    return A, 0.0

def init_pISTA_OBN_parser(pISTA_OBN_pasrser):
    pISTA_OBN_pasrser.set_defaults(algo='pISTA_OBN')
    pISTA_OBN_pasrser.add_argument(
        '-T', '--T', required=False, type=int, default=15, dest='T',
        help="Number of iterations.")
    pISTA_OBN_pasrser.add_argument(
        '-inner_T', '--inner_T', required=False, type=int, default=50, dest='inner_T',
        help="Number of inner iterations.")
    pISTA_OBN_pasrser.add_argument(
        '-linesearch', '--linesearch', required=False, type=int, default=15, dest='ls_iter',
        help="Number of linesearch iterations.")
    pISTA_OBN_pasrser.add_argument(
        '-step_lim', '--step_limit', required=False, type=float, default=1e-4, dest='step_lim',
        help="The smallest step size possible.")
    pISTA_OBN_pasrser.add_argument(
        '-st', '--step', required=False, type=float, default=1.0, dest='init_step',
        help='init_step.')