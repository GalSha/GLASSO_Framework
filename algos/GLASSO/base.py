class base(object):
    def __init__ (self, T, N, lam):
        self.T = T
        self.N = N
        self.lam = lam

    def compute(self, S, M, A0, status_f, history, test_check_f):
        pass

    def __call__(self, S, M):
        return self.compute(S, M, None, None, False, None)[0]

    def compute_full(self, S, M, status_f):
        return self.compute(S, M, None, status_f, True, None)[:-1]

    def compute_status(self, S, M, status_f):
        return self.compute(S, M, None, status_f, False, None)[:-2]

    def compute_final(self, S, M):
        return self(S, M)

    def compute_test(self, S, M, test_check_f):
        res = self.compute(S, M, None, None, False, test_check_f)
        return res[0], res[-1]

    def compute_warmup(self, S, M):
        T = self.T
        self.T = 1
        self.compute(S, M, None, None, False, None)
        self.T = T

    def name(self):
        pass