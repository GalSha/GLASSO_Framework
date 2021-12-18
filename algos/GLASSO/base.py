class base(object):
    def __init__ (self, T, N, lam):
        self.T = T
        self.N = N
        self.lam = lam

    def compute(self, S, A0, status_f, history, test_check_f):
        pass

    def __call__(self, S):
        return self.compute(S, None, None, False, None)[0]

    def compute_full(self, S, status_f):
        return self.compute(S, None, status_f, True, None)[:-1]

    def compute_status(self, S, status_f):
        return self.compute(S, None, status_f, False, None)[:-2]

    def compute_final(self, S):
        return self(S)

    def compute_test(self, S, test_check_f):
        res = self.compute(S, None, None, False, test_check_f)
        return res[0], res[-1]

    def compute_warmup(self, S):
        T = self.T
        self.T = 1
        self.compute(S, None, None, False, None)
        self.T = T

    def name(self):
        pass