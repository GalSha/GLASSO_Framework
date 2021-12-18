from numpy import inf
from numpy import linalg
from numpy import sqrt
from algos.GLASSO.base import base
from utils.common import cp_soft_threshold
from utils.GLASSO.glasso import cuda_objective_f_cholesky

class cuda_NL_fista(base):
    def __init__(self, T,  N, lam, inner_T, ls_iter, step_lim):
        super(cuda_NL_fista, self).__init__(T, N, lam)
        self.inner_T = inner_T
        self.ls_iter = ls_iter
        self.step_lim = step_lim
        self.save_name = "cuda_NL_fista_N{N}_T{T}_innerT{inner_T}_LsIter{ls_iter}_StepLim{step_lim}" \
            .format(N=self.N, T=self.T, inner_T=self.inner_T, ls_iter=self.ls_iter, step_lim=self.step_lim)

    def compute(self, S, A0, status_f, history, test_check_f):
        import cupy as cp
        import cupyx
        cupyx.seterr(linalg='raise')
        S = cp.array(S, dtype='float32')
        As = []
        status = []
        cp_step_lim = cp.float32(self.step_lim)
        lam = cp.float32(self.lam)
        init_step = cp.float32(1.0)

        if A0 is None:
            A_diag = self.lam * cp.ones(self.N, dtype='float32')
            A_diag = A_diag + cp.diag(S)
            A_diag = 1.0 / A_diag
            A = cp.diag(A_diag)
            A_diag = None
        else:
            A = cp.array(A0, dtype='float32')

        if history:
            As.append(cp.asnumpy(A))

        if status_f is not None: status.append(status_f(A, 0.0))

        for t in range(self.T):
            A_inv = cp.linalg.inv(A)
            if test_check_f is not None:
                if test_check_f(A, S, self.lam, A_inv):
                    break

            sign_A = cp.sign(A, dtype='float32')
            mask_A = cp.abs(sign_A, dtype='float32').astype('int8')
            G = S - A_inv
            F_subgrad_norm = cp.linalg.norm(cp_soft_threshold(cp, G + lam*sign_A, lam*(1.0-mask_A)))
            sign_A = None
            mask_G = cp.abs(cp.sign(cp_soft_threshold(cp, G, lam), dtype='float32')).astype('int8')
            mask = cp.bitwise_or(mask_A, mask_G)
            mask_G = None
            mask_A = None
            G_A_inv = G - A_inv

            inner_A = A
            inv_in_inv = A
            step = cp.linalg.eigvalsh(A)[0] ** 2
            step = 1 / step
            t_k = 1
            for inner_t in range(self.inner_T):
                if init_step == 0: break
                a = G_A_inv + inv_in_inv
                inv_in_inv = None
                inner_A_next = mask * cp_soft_threshold(cp, inner_A - (1 / step) * a, lam / step)
                t_k_next = cp.float32(0.5 * (1 + sqrt(1+4*t_k*t_k)))
                inner_A = inner_A_next + (t_k - 1) / t_k_next * (inner_A_next - inner_A)
                t_k = t_k_next
                a = None
                inv_in_inv = A_inv@inner_A@A_inv
                sign_inner_A = cp.sign(inner_A, dtype='float32')
                mask_inner_A = cp.abs(sign_inner_A, dtype='float32').astype('int8')
                inner_f_subgrad_min = cp.linalg.norm(cp_soft_threshold(cp, G + inv_in_inv + lam*sign_inner_A, lam*mask_inner_A))
                sign_inner_A = None
                mask_inner_A = None
                if inner_f_subgrad_min < 0.1 * F_subgrad_norm: break

            A, step = armijo_linesearch_F(cp, A, S, lam, G, A_inv, inner_A - A, init=init_step, max_iter=self.ls_iter, step_lim=cp_step_lim)
            if step == 0: init_step = 0

            if history:
                As.append(cp.asnumpy(A))

            if status_f is not None: status.append(status_f(A, step))

        if init_step == 0: t = inf
        return A, status, As, t+1

def objective_Q(cp, A, A_next, G, A_inv):
    A_next_A = A_next - A
    return cp.trace(A_next_A @ G, dtype='float32') +\
                0.5 * (cp.sum(cp.square(A_next_A@A_inv, dtype='float32'), dtype='float32'))

def armijo_linesearch_F(cp, A, S, lam, g, A_inv, Delta, init = 1, beta = 0.5, c = 0.1 , max_iter=10, step_lim=0):
    L = cp.linalg.cholesky(A)
    init_F_val = cuda_objective_f_cholesky(cp,A,S,L)
    step = init
    for _ in range(max_iter):
        if step < step_lim: break
        try:
            A_next = A+step*Delta
            A_next = A_next + cp.transpose(A_next)
            A_next *= 0.5
            L_next = cp.linalg.cholesky(A_next)
            if cuda_objective_f_cholesky(cp,A_next,S,L_next) <= init_F_val + step*c*objective_Q(cp,A, A_next, g, A_inv):
                    return A_next, step
        except linalg.LinAlgError:
            pass
        step *= beta
    return A, 0.0
