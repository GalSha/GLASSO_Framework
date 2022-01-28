from numpy import inf
from numpy import linalg
from algos.GLASSO.base import base
from utils.common import cp_soft_threshold
from utils.GLASSO.glasso import cuda_objective_F_cholesky
from algos.GLASSO.cuda_GISTA import cuda_GISTA_linesearch

class cuda_pISTA_GISTA(base):
    def __init__(self, T, N, lam, ls_iter, step_lim, init_step):
        super(cuda_pISTA_GISTA, self).__init__(T, N, lam)
        self.ls_iter = ls_iter
        self.step_lim = step_lim
        self.init_step = init_step
        self.save_name = "cuda_pISTA_GISTA_N{N}_T{T}_step{step}_LsIter{ls_iter}_StepLim{step_lim}"\
            .format(N=self.N, T=self.T, step=self.init_step, ls_iter=self.ls_iter, step_lim=self.step_lim)

    def compute(self, S, A0, status_f, history, test_check_f):
        import cupy as cp
        import cupyx as cpx
        import cupyx.scipy.sparse as cp_sp
        import cupyx.scipy.sparse.linalg as cp_spl
        cpx.seterr(linalg='raise')
        init_step = cp.float32(self.init_step)
        S = cp.array(S, dtype='float32')
        As = []
        status = []
        cp_step_lim = cp.float32(self.step_lim)

        lam = cp.float32(self.lam)

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
                    t -= 1
                    break
            if t % 2 == 0:
                sign_A = cp.sign(A, dtype='float32')
                mask_A = cp.abs(sign_A, dtype='int8')
                G = S - A_inv
                sign_soft_G = cp.sign(cp_soft_threshold(cp, G, lam),dtype='float32')
                mask_G = cp.abs(sign_soft_G,dtype='int8')
                mask = cp.bitwise_or(mask_A, mask_G)
                mask_G = None

                AgA = A @ (mask * G) @ A
                G = None

                sign_A -= cp.bitwise_xor(mask, mask_A) * sign_soft_G
                sign_soft_G = None

                AhA = A @ sign_A @ A
                AhA *= lam
                A_diag = cp.diag(A).copy().reshape(-1,1)
                cp.fill_diagonal(A, 0)
                A_no_diag = A
                AAt = ((A_no_diag * A_no_diag) + (A_diag * A_diag.T)) * sign_A
                AAt *= lam
                cp.fill_diagonal(A, A_diag)
                A_diag = None
                A_no_diag = None
                sign_A = None

                AghA = AgA + AhA - AAt
                AgA = None
                AhA = None

                A, step = pista_cholesky_linesearch(cp, A, S, lam, mask, AghA, AAt, A,
                                                            step=init_step, max_iter=self.ls_iter, step_lim=cp_step_lim)

                #if step == 0: init_step = 0
            else:
                A, step = cuda_GISTA_linesearch(cp, A, S, lam, A_inv, max_iter=self.ls_iter, init_step=init_step, step_lim=cp_step_lim)


            if history:
                As.append(cp.asnumpy(A))

            if status_f is not None: status.append(status_f(A, step))

        if init_step == 0: t = inf
        return A, status, As, t+1

def pista_cholesky_linesearch(cp, A, S, lam, mask, a, b, c, step, max_iter, step_lim):
    if step == 0:
        return A, 0.0
    beta = step
    L = cp.linalg.cholesky(A)
    init_F_value = cuda_objective_F_cholesky(cp, A,S,lam,L)
    L = None
    beta_psd = None
    for _ in range(max_iter):
        if beta < step_lim: break
        try:
            beta_a = beta * a
            beta_b = cp.abs(beta * b, dtype='float32')
            A_next = mask * cp_soft_threshold(cp, c - beta_a, beta_b)
            beta_a = None
            beta_b = None

            A_next = A_next + cp.transpose(A_next)
            A_next *= 0.5
            L = cp.linalg.cholesky(A_next)
            if beta_psd is None: beta_psd = beta
            if cuda_objective_F_cholesky(cp, A_next,S,lam,L) < init_F_value:
                return A_next, beta
        except linalg.LinAlgError:
            pass
        beta *= 0.5

    eigs = cp.linalg.eigvalsh(A)
    beta = (eigs[0]/eigs[-1]) ** 2
    beta = cp.float32(cp.asnumpy(0.81 * beta))
    eigs = None
    if beta_psd is not None and beta > beta_psd: beta = beta_psd
    beta_eps = cp.finfo(cp.float32).eps
    while True:
        try:
            beta_a = beta * a
            beta_b = cp.abs(beta * b, dtype='float32')
            A_next = mask * cp_soft_threshold(cp, c - beta_a, beta_b)
            beta_a = None
            beta_b = None

            A_next = A_next + cp.transpose(A_next)
            A_next *= 0.5
            L = cp.linalg.cholesky(A_next)
            return A_next, beta
        except linalg.LinAlgError:
            pass
        beta *= 0.5
        #Emulate do while
        if beta < beta_eps: break

    return A, 0.0