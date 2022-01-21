from numpy import inf
from numpy import linalg
from algos.GLASSO.base import base
from utils.common import cp_soft_threshold
from utils.GLASSO.glasso import cuda_objective_F_cholesky

class cuda_pISTA(base):
    def __init__(self, T, N, lam, ls_iter, step_lim, init_step, c_ls, proj, st):
        super(cuda_pISTA, self).__init__(T, N, lam)
        self.ls_iter = ls_iter
        self.step_lim = step_lim
        self.init_step = init_step
        self.c_ls = c_ls
        self.proj = proj
        proj_str = ""
        if self.proj: proj_str = "_P"
        self.st = None
        if st:
            import cupy
            self.st = cupy.ElementwiseKernel('float32 x, float32 lam', 'float32 y',
                                      'float xa = fabs(x); y = signbit(lam-xa)*copysign(xa-lam,x)', 'st')
        self.save_name = "cuda_pISTA_N{N}_T{T}_step{step}_LsIter{ls_iter}_StepLim{step_lim}_C{c_ls}{proj_str}"\
            .format(N=self.N, T=self.T, step=self.init_step, ls_iter=self.ls_iter, step_lim=self.step_lim, c_ls=self.c_ls,
                    proj_str=proj_str)

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

            sign_A = cp.sign(A, dtype='float32')
            mask_A = cp.abs(sign_A, dtype='int8')
            G = S - A_inv
            G_min = cp_soft_threshold(cp, G + lam * sign_A, lam * (1.0 - mask_A))
            sign_soft_G = cp.sign(cp_soft_threshold(cp, G, lam),dtype='float32')
            mask_G = cp.abs(sign_soft_G,dtype='int8')
            mask = cp.bitwise_or(mask_A, mask_G)
            mask_G = None

            AgA = A @ (mask * G) @ A
            G = None

            sign_A_next = sign_A - cp.bitwise_xor(mask, mask_A) * sign_soft_G
            if not self.proj: sign_A = None
            sign_soft_G = None

            AhA = A @ sign_A_next @ A
            AhA *= lam
            A_diag = cp.diag(A).copy().reshape(-1,1)
            cp.fill_diagonal(A, 0)
            A_no_diag = A
            AAt = ((A_no_diag * A_no_diag) + (A_diag * A_diag.T)) * sign_A_next
            AAt *= lam
            cp.fill_diagonal(A, A_diag)
            A_diag = None
            A_no_diag = None
            sign_A_next = None

            AghA = AgA + AhA - AAt
            AgA = None
            AhA = None

            A, step = pista_cholesky_linesearch(cp, A, S, lam, mask, AghA, AAt, A, G_min, c_ls=self.c_ls,
                                                        step=init_step, max_iter=self.ls_iter, step_lim=cp_step_lim,
                                                        sign_A=sign_A, st=self.st)
            if step == 0: init_step = 0

            if history:
                As.append(cp.asnumpy(A))

            if status_f is not None: status.append(status_f(A, step))

        if init_step == 0: t = inf
        return A, status, As, t+1

def pista_cholesky_linesearch(cp, A, S, lam, mask, a, b, c, g_min, c_ls, step, max_iter, step_lim, sign_A, st):
    if step == 0:
        return A, 0.0
    beta = step
    L = cp.linalg.cholesky(A)
    c_ls = cp.float32(c_ls)
    init_F_val = cuda_objective_F_cholesky(cp, A, S, lam, L)
    init_F_val_gA = init_F_val - c_ls * cp.sum(g_min * A, dtype='float32')
    L = None
    beta_psd = None
    beta_dec = None
    for _ in range(max_iter):
        if beta < step_lim: break
        try:
            beta_a = beta * a
            beta_b = cp.abs(beta * b, dtype='float32')
            if st is None: A_next = mask * cp_soft_threshold(cp, c - beta_a, beta_b)
            else: A_next = mask * st(c - beta_a, beta_b)
            if sign_A is not None and cp.any(cp.sign(A_next, dtype='float32') * sign_A < 0):
                # if sign_A is not None and np.min(np.sign(A_next, dtype='float32') * sign_A) == -1:
                beta *= 0.5
                continue
                # raise linalg.LinAlgError
            beta_a = None
            beta_b = None

            A_next = A_next + cp.transpose(A_next)
            A_next *= 0.5
            L = cp.linalg.cholesky(A_next)
            if beta_psd is None: beta_psd = beta
            next_F_val = cuda_objective_F_cholesky(cp, A_next,S,lam,L)
            if c_ls > 0 and next_F_val <= init_F_val_gA + c_ls * cp.sum(g_min * A_next, dtype='float32'):
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
        beta_b = cp.abs(beta * b, dtype='float32')
        A_next = mask * cp_soft_threshold(cp, c - beta_a, beta_b)
        beta_a = None
        beta_b = None

        A_next = A_next + cp.transpose(A_next)
        A_next *= 0.5
        return A_next, beta

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