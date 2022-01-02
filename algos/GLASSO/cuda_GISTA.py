from numpy import inf
from numpy import linalg
from algos.GLASSO.base import base
from utils.common import cp_soft_threshold
from utils.GLASSO.glasso import cuda_objective_f_cholesky

class cuda_GISTA(base):
    def __init__(self, T, N, lam, ls_iter, step_lim):
        super(cuda_GISTA, self).__init__(T, N, lam)
        self.ls_iter = ls_iter
        self.step_lim = step_lim
        self.save_name = "cuda_GISTA_N{N}_T{T}_LsIter{ls_iter}_StepLim{step_lim}" \
            .format(N=self.N, T=self.T, ls_iter=self.ls_iter, step_lim=self.step_lim)

    def compute(self, S, A0, status_f, history, test_check_f):
        import cupy as cp
        import cupyx
        cupyx.seterr(linalg='raise')
        S = cp.array(S, dtype='float32')
        As = []
        status = []
        cp_step_lim = cp.float32(self.step_lim)

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

        if status_f is not None: status.append(status_f(A, cp.asarray(0.0)))

        init_step = cp.asarray(1.0, dtype='float32')
        A_inv = cp.linalg.inv(A)
        for t in range(self.T):
            if test_check_f is not None:
                if test_check_f(A, S, self.lam, A_inv):
                    t -= 1
                    break

            A_next, step = cuda_GISTA_linesearch(cp, A, S, self.lam, A_inv, max_iter=self.ls_iter, init_step=init_step, step_lim=cp_step_lim)
            if step == 0:
               init_step = 0
            else:
                A_next_inv = cp.linalg.inv(A_next)

                A_next_A = A_next - A
                init_step = cp.sum(cp.square(A_next_A, dtype='float32'), dtype='float32')
                div_init_step = cp.trace((A_next_A) @ (A_inv - A_next_inv), dtype='float32')
                A_next_A = None
                if div_init_step != 0:
                    init_step /= div_init_step
                else:
                    init_step = cp.asarray(0.0)
                A = A_next
                A_next = None
                A_inv = A_next_inv
                A_next_inv = None

            if history:
                As.append(cp.asnumpy(A))

            if status_f is not None: status.append(status_f(A, step))

        if init_step == 0: t = inf
        return A, status, As, t+1


def objective_Q(cp, objective_f_value, A, D, A_next, step):
    A_next_A = A_next - A
    return objective_f_value + cp.trace(A_next_A @ D, dtype='float32') + (
                0.5 / step) * (cp.sum(cp.square(A_next_A, dtype='float32'), dtype='float32'))


def cuda_GISTA_linesearch(cp, A, S, lam, A_inv, max_iter, init_step, step_lim):
    if init_step == 0:
        return A, cp.array(0.0)
    step = init_step
    D = S - A_inv
    L = cp.linalg.cholesky(A)
    init_F_value = cuda_objective_f_cholesky(cp, A,S,L)
    L = None
    for _ in range(max_iter):
        if step < step_lim: break
        try:
            A_next = cp_soft_threshold(cp, A - step * D, step * lam)
            A_next = A_next + cp.transpose(A_next)
            A_next *= 0.5
            L_next = cp.linalg.cholesky(A_next)
            if cuda_objective_f_cholesky(cp, A_next, S, L_next) <= objective_Q(cp, init_F_value, A, D, A_next, step):
                return A_next, step
        except linalg.LinAlgError:
            pass
        step *= 0.5

    step = cp.linalg.eigvalsh(A)[0] ** 2
    A_next = cp_soft_threshold(cp, A - step * D, step * lam)
    A_next = A_next + cp.transpose(A_next)
    A_next *= 0.5

    try:
        # TODO: not SPD sometimes
        L_next = cp.linalg.cholesky(A_next)
    except linalg.LinAlgError:
        step = 0.0
        A_next = A
    return A_next, cp.array(step)