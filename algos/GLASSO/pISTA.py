import numpy as np
from numpy import linalg
from algos.GLASSO.base import base
from utils.common import np_soft_threshold
from utils.GLASSO.glasso import objective_F_cholesky

class pISTA(base):
    def __init__(self, T, N, lam, ls_iter, step_lim, init_step, c_ls, proj, st):
        super(pISTA, self).__init__(T, N, lam)
        self.ls_iter = ls_iter
        self.step_lim = step_lim
        self.init_step = init_step
        self.c_ls = c_ls
        self.proj = proj
        proj_str = ""
        if self.proj: proj_str = "_P"
        self.save_name = "pISTA_N{N}_T{T}_step{step}_LsIter{ls_iter}_StepLim{step_lim}_C{c_ls}{proj_str}"\
            .format(N=self.N, T=self.T, step=self.init_step, ls_iter=self.ls_iter, step_lim=self.step_lim, c_ls=self.c_ls,
                    proj_str=proj_str)

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

            sign_A = np.sign(A, dtype='float32')
            mask_A = np.abs(sign_A, dtype='float32').astype('int8')
            G = S - A_inv
            G_min = np_soft_threshold(G + lam * sign_A, lam * (1.0 - mask_A))
            sign_soft_G = np.sign(np_soft_threshold(G, lam), dtype='float32')
            mask_G = np.abs(sign_soft_G).astype('int8')
            mask = np.bitwise_or(mask_A, mask_G)
            mask_G = None

            AgA = A @ (mask * G) @ A
            G = None

            sign_A_next = sign_A - np.bitwise_xor(mask, mask_A) * sign_soft_G
            if not self.proj: sign_A = None
            sign_soft_G = None

            AhA = A @ sign_A_next @ A
            AhA *= lam
            A_diag = np.diag(A).copy().reshape(-1, 1)
            np.fill_diagonal(A, 0)
            A_no_diag = A
            AAt = ((A_no_diag * A_no_diag) + (A_diag * A_diag.T)) * sign_A_next
            AAt *= lam
            np.fill_diagonal(A, A_diag)
            A_diag = None
            A_no_diag = None
            sign_A_next = None

            AghA = AgA + AhA - AAt
            AgA = None
            AhA = None

            A, step = pista_cholesky_linesearch(A, S, lam, mask, AghA, AAt, A, G_min, c_ls=self.c_ls,
                                                        step=init_step, max_iter=self.ls_iter, step_lim=self.step_lim,
                                                        sign_A=sign_A)
            if step == 0: init_step = 0

            if history:
                As.append(A.copy())

            if status_f is not None: status.append(status_f(A, step))

        if init_step == 0: t = np.inf
        return A, status, As, t+1

def pista_cholesky_linesearch(A, S, lam, mask, a, b, c, g_min, c_ls, step, max_iter, step_lim, sign_A):
    if step == 0:
        return A, 0.0
    beta = step
    L = np.linalg.cholesky(A)
    init_F_val = objective_F_cholesky(A,S,lam,L)
    init_F_val_gA = init_F_val - c_ls*np.sum(g_min*A, dtype='float32')
    L = None
    beta_psd = None
    beta_dec = None
    for _ in range(max_iter):
        if beta < step_lim: break
        try:
            beta_a = beta * a
            beta_b = np.abs(beta * b, dtype='float32')
            A_next = mask * np_soft_threshold(c - beta_a, beta_b)
            if sign_A is not None and np.any(np.sign(A_next, dtype='float32') * sign_A < 0):
            #if sign_A is not None and np.min(np.sign(A_next, dtype='float32') * sign_A) == -1:
                beta *= 0.5
                continue
                #raise linalg.LinAlgError
            beta_a = None
            beta_b = None

            A_next = A_next + np.transpose(A_next)
            A_next *= 0.5
            L = np.linalg.cholesky(A_next)
            if beta_psd is None: beta_psd = beta
            next_F_val = objective_F_cholesky(A_next,S,lam,L)
            if c_ls > 0 and next_F_val <= init_F_val_gA + c_ls*np.sum(g_min*A_next, dtype='float32'):
                return A_next, beta
            if next_F_val < init_F_val:
                if c_ls <= 0: return A_next, beta
                if beta_dec is None: beta_dec = beta
        except linalg.LinAlgError:
            pass
        beta *= 0.5

    if beta_dec is not None:
        beta = beta_dec
        beta_a = beta * a
        beta_b = np.abs(beta * b, dtype='float32')
        A_next = mask * np_soft_threshold(c - beta_a, beta_b)
        beta_a = None
        beta_b = None

        A_next = A_next + np.transpose(A_next)
        A_next *= 0.5
        return A_next, beta

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
        '-c', '--c', required=False, type=float, default=0, dest='c_ls',
        help='c for linesearch.')
    pISTA_pasrser.add_argument(
        '-p', '--proj', action='store_true', required=False, default=False, dest='proj',
        help='proj check.')
    pISTA_pasrser.add_argument(
        '-u', '--u', action='store_true', required=False, default=False, dest='st',
        help='user st.')