from numpy import maximum
from algos.GLASSO.base import base
from utils.common import cp_soft_threshold
from utils.common import cp_hard_threshold


class cuda_ALM(base):
    def __init__(self, T, N, lam, N_mu, eta, skip, step_lim):
        super(cuda_ALM, self).__init__(T, N, lam)
        import numpy as np
        self.N_mu = N_mu
        self.eta = eta
        self.skip = skip
        skip_str = ""
        if self.skip: skip_str="skip"
        if self.lam < 0.5:
            #self.mu_0 = np.float32(100 / self.lam)
            self.mu_0 = np.float32(1 / self.lam)
        elif self.lam < 10:
            self.mu_0 = self.lam
        else:
            self.mu_0 = self.lam / 100
        self.min_mu = np.maximum(1e-6, self.mu_0 * ((1/self.eta) ** 8), dtype='float32')
        self.step_lim = step_lim
        self.save_name = "cuda_ALM{skip}_N{N}_T{T}_Nmu{N_mu}_eta{eta}_StepLim{step_lim}"\
            .format(skip=skip_str, N=self.N, T=self.T, N_mu=self.N_mu, eta=self.eta, step_lim=self.step_lim)

    def compute(self, S, A0, status_f, history, test_check_f):
        import cupy as cp
        import cupyx
        cupyx.seterr(linalg='raise')
        S = cp.array(S, dtype='float32')
        As = []
        status = []
        lam = cp.float32(self.lam)
        eta = cp.float32(self.eta)
        min_mu = self.min_mu
        eps = 10*cp.float32(cp.finfo(cp.float32).eps)

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

        X = A
        Y = A
        A = None

        alpha_2 = 1 / (2 * (cp.linalg.norm(S, ord=2) + lam * S.shape[0]))
        mu = cp.float32(self.mu_0)
        X_inv = cp.linalg.inv(X)
        #G = S - X_inv
        for t in range(self.T):
            if test_check_f is not None:
                if test_check_f(X, S, self.lam, X_inv):
                    break
            #Lam = G - (X - Y)/mu
            if t > 0 and t % self.N_mu == 0: mu = cp.float32(maximum(mu / eta, min_mu))
            #d, V = cp.linalg.eigh(Y + mu*(Lam - S))
            d, V = cp.linalg.eigh(2*Y - mu*X_inv - X)

            gamma = d + cp.sqrt(d*d + 4*mu)
            gamma = cp.maximum(gamma/2, alpha_2)

            X = V @ cp.diag(gamma) @ V.T
            X = cp_hard_threshold(cp, X, eps)
            X_inv = V @ cp.diag(1 / gamma) @ V.T

            G = S - X_inv
            Y = cp_soft_threshold(cp, X - mu * G, mu*lam)
            G = None

            if history:
                As.append(cp.asnumpy(X))

            if status_f is not None: status.append(status_f(X, mu))

        return X, status, As, t+1