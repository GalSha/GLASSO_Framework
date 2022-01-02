from numpy import linalg, inf
from algos.GLASSO.base import base
from utils.common import cp_soft_threshold
from utils.GLASSO.glasso import cuda_objective_F_cholesky, objective_g

class cuda_QUIC(base):
    def __init__(self, T, N, lam, inner_T, armijo_iter, step_lim):
        super(cuda_QUIC, self).__init__(T,N,lam)
        self.inner_T = inner_T
        self.armijo_iter = armijo_iter
        self.step_lim = step_lim
        self.save_name = "cuda_QUIC_N{N}_T{T}_innerT{inner_T}_armijoIter{armijo_iter}_StepLim{step_lim}"\
            .format(N=self.N, T=self.T, inner_T=self.inner_T, armijo_iter=self.armijo_iter,
                    step_lim=self.step_lim)

    def compute(self, S, M, A0, status_f, history, test_check_f):
        import cupy as cp
        import cupyx
        cupyx.seterr(linalg='raise')
        S = cp.array(S, dtype='float32')
        M = cp.array(M, dtype='int8')
        As = []
        status = []
        not_I = 1 - cp.eye(self.N, dtype='int8')
        cp_lam = cp.float32(self.lam)
        cp_step_lim = cp.float32(self.step_lim)
        init_step = cp.float32(1.0)

        if A0 is None:
            A_diag = self.lam * cp.ones(self.N, dtype='float32')
            A_diag = A_diag + cp.diag(S)
            A_diag = 1.0 / A_diag
            A = cp.diag(A_diag)
            A_diag = None
        else:
            A = cp.array(A0, dtype='float32')
        A = M * A

        if history:
            As.append(cp.asnumpy(A))

        if status_f is not None: status.append(status_f(A, cp.asarray(0.0)))

        step = cp.float32(1.0)
        for t in range(self.T):
            W = cp.linalg.inv(A)
            if test_check_f is not None:
                if test_check_f(A, S, self.lam, W):
                    t -= 1
                    break

            if init_step != 0:
                g = objective_g(W, S)

                W_diag = cp.diag(W).reshape(-1, 1)
                M = not_I * (W ** 2) + W_diag @ W_diag.T
                W_diag = None
                div_M = 1.0 / M
                M = None
                lam_div_M = cp_lam * div_M

                D = cp.zeros((self.N, self.N), dtype='float32')
                U = cp.zeros((self.N, self.N), dtype='float32')
                inner_T = self.inner_T
                if inner_T < 0: inner_T = int(1 - t / inner_T)
                for _ in range(inner_T):
                    for i in range(self.N):
                        for j in range(i+1):
                            if M[i,j] == 0: continue
                            b = g[i,j] + W[:,i]@U[:,j]
                            b_div_M = b * div_M[i,j]
                            mu = cp_soft_threshold(cp,A[i,j] + D[i,j] - b_div_M, lam_div_M[i,j]) - A[i,j] - D[i,j]
                            D[i,j] += mu
                            U[j,:] = U[j,:] + mu * W[i,:]
                            if j != i:
                                D[j,i] += mu
                                U[i, :] = U[i, :] + mu * W[j, :]

                A, step = armijo_linesearch_F(cp, A, S, cp_lam, g, D, max_iter=self.armijo_iter, step_lim=cp_step_lim, init=init_step)
                if step == 0: init_step = 0

            if history:
                As.append(cp.asnumpy(A))

            if status_f is not None: status.append(status_f(A, step))

        if init_step == 0: t = inf
        return A, status, As, t+1

def armijo_linesearch_F(cp, A, S, lam, g, Delta, init = 1, beta = 0.5, c = 0.1 , max_iter=10, step_lim=0):
    L = cp.linalg.cholesky(A)
    init_F_val = cuda_objective_F_cholesky(cp, A,S,lam,L)
    L = None
    step = init
    g_Delta_trace = cp.trace(g@Delta, dtype='float32')
    lam_term = lam*cp.sum((cp.abs(A+Delta, dtype='float32')-cp.abs(A, dtype='float32')), dtype='float32')
    g_Delta_trace_lam_term = g_Delta_trace+lam_term
    for _ in range(max_iter):
        if step < step_lim: break
        try:
            A_next = A+step*Delta
            A_next = A_next + cp.transpose(A_next)
            A_next *= 0.5
            L_next = cp.linalg.cholesky(A_next)
            if cuda_objective_F_cholesky(cp, A_next,S,lam,L_next) <= init_F_val + step*c*(g_Delta_trace_lam_term):
                    return A_next, cp.array(step)
        except linalg.LinAlgError:
            pass
        step *= beta
    return A, cp.array(0.0)
