import numpy as np
from numpy import linalg
from algos.GLASSO.base import base
from utils.common import np_soft_threshold
from utils.GLASSO.glasso import objective_f_cholesky

class GISTA(base):
    def __init__(self, T, N, lam, ls_iter, step_lim):
        super(GISTA, self).__init__(T, N, lam)
        self.ls_iter = ls_iter
        self.step_lim = step_lim
        self.save_name = "GISTA_N{N}_T{T}_LsIter{ls_iter}_StepLim{step_lim}"\
            .format(N=self.N, T=self.T, ls_iter=self.ls_iter, step_lim=self.step_lim)

    def compute(self, S, M, A0, status_f, history, test_check_f):
        As = []
        status = []

        if A0 is None:
            A_diag = self.lam*np.ones(self.N, dtype='float32')
            A_diag = A_diag + np.diag(S)
            A_diag = 1.0 / A_diag
            A = np.diag(A_diag)
        else:
            A = np.array(A0, dtype='float32')
        if history:
            As.append(A.copy())
        A = M * A

        if status_f is not None: status.append(status_f(A, 0.0))

        init_step = np.float32(1.0)
        A_inv = np.linalg.inv(A)
        for t in range(self.T):
            if test_check_f is not None:
                if test_check_f(A, S, self.lam, A_inv):
                    t -= 1
                    break

            A_next, step = GISTA_linesearch(A, S, M, self.lam, A_inv, max_iter=self.ls_iter, init_step=init_step,
                                            step_lim=self.step_lim)
            if step == 0:
               init_step = 0
            else:
                A_next_inv = np.linalg.inv(A_next)

                A_next_A = A_next - A
                init_step = np.sum(np.square(A_next_A, dtype='float32'), dtype='float32')
                div_init_step = np.trace((A_next_A) @ (A_inv - A_next_inv), dtype='float32')
                A_next_A = None
                if div_init_step != 0:
                    init_step /= div_init_step
                else:
                    init_step = 0
                A = A_next
                A_next = None
                A_inv = A_next_inv
                A_next_inv = None

            if history:
                As.append(A.copy())

            if status_f is not None: status.append(status_f(A, step))

        return A, status, As, t+1

def objective_Q(objective_f_value, A, D, A_next, step):
    A_next_A = A_next - A
    return objective_f_value + np.trace(A_next_A @ D, dtype='float32') + (
                0.5 / step) * (np.sum(np.square(A_next_A, dtype='float32'), dtype='float32'))

def GISTA_linesearch(A, S, M, lam, A_inv, max_iter, init_step, step_lim):
    if init_step == 0:
        return A, 0.0
    step = init_step
    D = S - A_inv
    D = M * D
    L = np.linalg.cholesky(A)
    init_F_value = objective_f_cholesky(A,S,L)
    L = None
    for _ in range(max_iter):
        if step < step_lim: break
        try:
            A_next = np_soft_threshold(A- step * D, step * lam)
            A_next = A_next + np.transpose(A_next)
            A_next *= 0.5
            L_next = np.linalg.cholesky(A_next)
            if objective_f_cholesky(A_next, S, L_next) <= objective_Q(init_F_value, A, D, A_next, step):
                return A_next, step
        except linalg.LinAlgError:
            pass
        step *= 0.5
    step = np.linalg.eigvalsh(A)[0] ** 2
    A_next = np_soft_threshold(A - step * D, step * lam)
    A_next = A_next + np.transpose(A_next)
    A_next *= 0.5
    try:
        L_next = np.linalg.cholesky(A_next)
    except linalg.LinAlgError:
        step = 0.0
        A_next = A
    return A_next, step

def init_GISTA_parser(GISTA_pasrser):
    GISTA_pasrser.set_defaults(algo='GISTA')
    GISTA_pasrser.add_argument(
        '-T', '--T', required=False, type=int, default=15, dest='T',
        help="Number of iterations.")
    GISTA_pasrser.add_argument(
        '-linesearch', '--linesearch', required=False, type=int, default=15, dest='ls_iter',
        help="Number of linesearch iterations.")
    GISTA_pasrser.add_argument(
        '-step_lim', '--step_limit', required=False, type=float, default=1e-4, dest='step_lim',
        help="The smallest step size possible.")
