import numpy as np
from algos.GLASSO.base import base
from utils.common import np_soft_threshold
from utils.common import np_hard_threshold

class ALM(base):
    def __init__(self, T, N, lam, N_mu, eta, skip, step_lim):
        super(ALM, self).__init__(T, N, lam)
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
        self.save_name = "ALM{skip}_N{N}_T{T}_Nmu{N_mu}_eta{eta}_StepLim{step_lim}"\
            .format(skip=skip_str, N=self.N, T=self.T, N_mu=self.N_mu, eta=self.eta, step_lim=self.step_lim)

    def compute(self, S, M, A0, status_f, history, test_check_f):
        eps = 10*np.float32(np.finfo(np.float32).eps)
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

        X = A
        Y = A
        A = None

        alpha_2 = 1 / (2 * (np.linalg.norm(S, ord=2) + self.lam * S.shape[0]))
        mu = self.mu_0
        X_inv = np.linalg.inv(X)
        G = S - X_inv
        skip_check = False
        for t in range(self.T):
            if not skip_check:
                if test_check_f is not None:
                    if test_check_f(X, S, self.lam, X_inv):
                        t -= 1
                        break
            Lam = G - (X - Y)/mu
            if t > 0 and t % self.N_mu == 0: mu = np.maximum(mu / self.eta, self.min_mu)
            #d, V = np.linalg.eigh(Y + mu*(Lam - S))
            d, V = np.linalg.eigh(2*Y - mu*X_inv - X)

            gamma = d + np.sqrt(d*d + 4*mu)
            gamma = np.maximum(gamma/2, alpha_2)

            if not skip_check: X_old = X
            X = V @ np.diag(gamma) @ V.T
            #X = M * np_hard_threshold(X, eps)
            X = np_hard_threshold(X, eps)

            X_Y = X - Y
            if self.skip:
                if self.lam * np.sum(np.abs(X, dtype='float32'), dtype='float32') > \
                    self.lam * np.sum(np.abs(Y, dtype='float32'), dtype='float32') \
                        - np.sum(Lam * X_Y, dtype='float32') + (1 / (2 * mu)) * np.sum(X_Y * X_Y, dtype='float32'):
                    skip_check = True
                    X = Y
                    d, V = np.linalg.eigh(X)
                    gamma = np.maximum(d, alpha_2)
                    X = V @ np.diag(gamma) @ V.T
                    #X = M * np_hard_threshold(X, eps)
                    X = np_hard_threshold(X, eps)
                    X_inv = V @ np.diag(1 / gamma) @ V.T
                else:
                    skip_check = False
                    X_old = None
                    X_inv = V @ np.diag(1 / gamma) @ V.T
            else:
                X_old = None
                X_inv = V @ np.diag(1 / gamma) @ V.T

            G = S - X_inv
            Y = M * np_soft_threshold(X - mu * G, mu*self.lam)

            if history:
                if skip_check: As.append(X_old.copy())
                else: As.append(X.copy())

            if status_f is not None:
                if skip_check: status.append(status_f(X_old, mu))
                else: status.append(status_f(X, mu))

        return X, status, As, t+1

def init_ALM_parser(ALM_pasrser):
    ALM_pasrser.set_defaults(algo='ALM')
    ALM_pasrser.add_argument(
        '-T', '--T', required=False, type=int, default=15, dest='T',
        help="Number of iterations.")
    ALM_pasrser.add_argument(
        '-N_mu', '--N_mu', required=False, type=int, default=20, dest='N_mu',
        help="N_mu parameter.")
    ALM_pasrser.add_argument(
        '-eta', '--eta', required=False, type=int, default=3, dest='eta',
        help="Eta parameter.")
    ALM_pasrser.add_argument(
        '-step_lim', '--step_limit', required=False, type=float, default=1e-4, dest='step_lim',
        help="The smallest step size possible.")
    ALM_pasrser.set_defaults(skip=False)
    #Todo: add support for skipping step in cuda_ALM
    #ALM_pasrser.add_argument(
    #    '-skip', '--skip', action='store_true', required=False, default=False, dest='skip',
    #    help='Use skipping step.')
