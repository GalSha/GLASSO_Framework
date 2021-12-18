import numpy as np
from numpy import linalg
from algos.GLASSO.base import base
from utils.common import np_soft_threshold
from utils.GLASSO.glasso import objective_F_cholesky, objective_g

class QUIC(base):
    def __init__(self, T, N, lam, inner_T, armijo_iter, step_lim):
        super(QUIC, self).__init__(T,N,lam)
        self.inner_T = inner_T
        self.armijo_iter = armijo_iter
        self.step_lim = step_lim
        self.save_name = "QUIC_N{N}_T{T}_innerT{inner_T}_armijoIter{armijo_iter}_StepLim{step_lim}"\
            .format(N=self.N, T=self.T, inner_T=self.inner_T, armijo_iter=self.armijo_iter, step_lim=self.step_lim)

    def compute(self, S, A0, status_f, history, test_check_f):
        As = []
        status = []
        not_I = 1 - np.eye(self.N, dtype='int8')

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
            W = np.linalg.inv(A)
            if test_check_f is not None:
                if test_check_f(A, S, self.lam, W):
                    break

            if init_step != 0:
                g = objective_g(W, S)

                W_diag = np.diag(W).reshape(-1, 1)
                M = not_I * (W ** 2) + W_diag @ W_diag.T
                div_M = 1.0 / M
                lam_div_M = self.lam * div_M

                D = np.zeros((self.N, self.N), dtype='float32')
                U = np.zeros((self.N, self.N), dtype='float32')

                inner_T = self.inner_T
                if inner_T < 0: inner_T = int(1-t/inner_T)
                for _ in range(inner_T):
                    for i in range(self.N):
                        for j in range(i+1):
                            if np.abs(g[i,j], dtype='float32') < self.lam and A[i,j] == 0: continue
                            b = g[i,j] + W[:,i]@U[:,j]
                            b_div_M = b * div_M[i,j]
                            mu = np_soft_threshold(A[i,j] + D[i,j] - b_div_M, lam_div_M[i,j]) - A[i,j] - D[i,j]
                            D[i,j] += mu
                            U[j,:] = U[j,:] + mu * W[i,:]
                            if j != i:
                                D[j,i] += mu
                                U[i, :] = U[i, :] + mu * W[j, :]


                A, step = armijo_linesearch_F(A, S, self.lam, g, D, max_iter=self.armijo_iter, step_lim=self.step_lim, init=init_step)
                if step == 0: init_step = 0

            if history:
                As.append(A.copy())

            if status_f is not None: status.append(status_f(A, step))

        if init_step == 0: t = np.inf
        return A, status, As, t+1

def armijo_linesearch_F(A, S, lam, g, Delta, init = 1, beta = 0.5, c = 0.1 , max_iter=10, step_lim=0):
    L = np.linalg.cholesky(A)
    init_F_val = objective_F_cholesky(A,S,lam,L)
    step = init
    g_Delta_trace = np.trace(g@Delta, dtype='float32')
    lam_term = lam*np.sum((np.abs(A+Delta, dtype='float32')-np.abs(A, dtype='float32')), dtype='float32')
    g_Delta_trace_lam_term = g_Delta_trace+lam_term
    for _ in range(max_iter):
        if step < step_lim: break
        try:
            A_next = A+step*Delta
            A_next = A_next + np.transpose(A_next)
            A_next *= 0.5
            L_next = np.linalg.cholesky(A_next)
            if objective_F_cholesky(A_next,S,lam,L_next) <= init_F_val + step*c*(g_Delta_trace_lam_term):
                    return A_next, step
        except linalg.LinAlgError:
            pass
        step *= beta
    return A, 0

def init_QUIC_parser(QUIC_pasrser):
    QUIC_pasrser.set_defaults(algo='QUIC')
    QUIC_pasrser.add_argument(
        '-T', '--T', required=False, type=int, default=15, dest='T',
        help="Number of iterations.")
    QUIC_pasrser.add_argument(
        '-inT', '--inner_T', required=False, type=int, default=1, dest='inner_T',
        help="Number of inner iterations.")
    QUIC_pasrser.add_argument(
        '-armijo', '--armijo', required=False, type=int, default=10, dest='armijo_iter',
        help="Number of armijo iterations.")
    QUIC_pasrser.add_argument(
        '-step_lim', '--step_limit', required=False, type=float, default=1e-4, dest='step_lim',
        help="The smallest step size possible.")
