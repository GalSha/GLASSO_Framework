import numpy as np
from numpy import linalg
from algos.GLASSO.base import base
from utils.common import np_soft_threshold, np_cholesky_inv, np_cholesky, np_is_diag
from utils.GLASSO.glasso import objective_F_cholesky, objective_g
from algos.GLASSO.QUIC import armijo_linesearch_F

class pISTA(base):
    def __init__(self, T, N, lam, ls_iter, step_lim, init_step, quic):
        super(pISTA, self).__init__(T, N, lam)
        self.ls_iter = ls_iter
        self.step_lim = step_lim
        self.init_step = init_step
        self.quic_sigma  = None
        quic_str = ""
        if quic is not None and 0.5 > quic > 0:
            self.quic_sigma = quic
            quic_str = "_QUIC{quic}".format(quic=self.quic_sigma)
        self.save_name = "pISTA_N{N}_T{T}_step{step}_LsIter{ls_iter}_StepLim{step_lim}{quic_str}"\
            .format(N=self.N, T=self.T, step=self.init_step, ls_iter=self.ls_iter, step_lim=self.step_lim, quic_str=quic_str)

    def compute(self, S, A0, status_f, history, test_check_f):
        init_step = np.float32(self.init_step)
        quic_sigma = np.float32(self.quic_sigma)
        init_t = 0
        As = []
        status = []

        lam = np.float32(self.lam)

        A_is_diag = False
        if A0 is None:
            A_diag = self.lam * np.ones(self.N, dtype='float32')
            A_diag = A_diag + np.diag(S)
            A_diag = 1.0 / A_diag
            A = np.diag(A_diag)
            A_diag = None
            A_is_diag = True
        else:
            A = np.array(A0, dtype='float32')

        if history:
            As.append(A.copy())

        if status_f is not None: status.append(status_f(A, 0.0))

        if self.quic_sigma is not None and (A_is_diag or np_is_diag(A)):
            init_t = 1
            A_inv_diag = 1 / np.diag(A)
            A_inv = np.diag(A_inv_diag)
            A_inv_diag = A_inv_diag.reshape(-1,1)
            if test_check_f is not None:
                if test_check_f(A, S, self.lam, A_inv):
                    return A, status, As, 0
            t = 0
            G = S - A_inv
            A_inv = None
            A_inv_A_inv = A_inv_diag @ A_inv_diag.T
            D = -A + np_soft_threshold(A - (G / A_inv_A_inv), lam / A_inv_A_inv)
            A, step = armijo_linesearch_F(A, S, self.lam, G, D, max_iter=self.ls_iter, step_lim=self.step_lim, init=init_step, c=quic_sigma)
            if step == 0: init_step = 0
            if history:
                As.append(A.copy())

            if status_f is not None: status.append(status_f(A, step))

        A_inv = np.linalg.inv(A)
        for t in range(init_t, self.T):
            if test_check_f is not None:
                if test_check_f(A, S, self.lam, A_inv):
                    t -= 1
                    break

            sign_A = np.sign(A, dtype='float16')
            mask_A = np.abs(sign_A, dtype='float16').astype('int8')
            G = S - A_inv
            sign_soft_G = np.sign(np_soft_threshold(G, lam), dtype='float16')
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

            A, step, A_inv = pista_cholesky_linesearch(A, S, lam, mask, AghA, AAt, A,
                                                        step=init_step, max_iter=self.ls_iter, step_lim=self.step_lim)
            if step == 0: init_step = 0

            if history:
                As.append(A.copy())

            if status_f is not None: status.append(status_f(A, step))

        if init_step == 0: t = np.inf
        return A, status, As, t+1

def pista_cholesky_linesearch(A, S, lam, mask, a, b, c, step, max_iter, step_lim):
    if step == 0:
        return A, 0.0
    beta = step
    L = np_cholesky(A)
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
            L = np_cholesky(A_next)
            if beta_psd is None: beta_psd = beta
            if objective_F_cholesky(A_next,S,lam,L) < init_F_value:
                return A_next, beta, np_cholesky_inv(L)
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
            L = np_cholesky(A_next)
            return A_next, beta, np_cholesky_inv(L)
        except linalg.LinAlgError:
            pass
        beta *= 0.5
        #Emulate do while
        if beta < beta_eps: break

    return A, 0.0

def init_pISTA_parser(pISTA_pasrser):
    pISTA_pasrser.set_defaults(algo='pISTA')
    pISTA_pasrser.add_argument(
        '-T', '--T', required=False, type=int, default=15, dest='T',
        help="Number of iterations.")
    pISTA_pasrser.add_argument(
        '-linesearch', '--linesearch', required=False, type=int, default=15, dest='ls_iter',
        help="Number of linesearch iterations.")
    pISTA_pasrser.add_argument(
        '-step_lim', '--step_limit', required=False, type=float, default=1e-4, dest='step_lim',
        help="The smallest step size possible.")
    pISTA_pasrser.add_argument(
        '-st', '--step', required=False, type=float, default=1.0, dest='init_step',
        help='init_step.')
    pISTA_pasrser.add_argument(
        '-64', '--64', required=False, action='store_true', default=False, dest='bit64',
        help='use 64 bit floating point.')
    pISTA_pasrser.add_argument(
        '-quic', '--quic', required=False, type=float, default=None, dest='quic',
        help='use diagonal QUIC iteration as first iteration.')