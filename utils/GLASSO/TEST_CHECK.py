import time
from utils.GLASSO.glasso import gap_test_check, cuda_gap_test_check
from utils.GLASSO.glasso import rel_test_check, cuda_rel_test_check
from utils.GLASSO.glasso import gap_rel_test_check, cuda_gap_rel_test_check
from utils.GLASSO.glasso import nmse_test_check, cuda_nmse_test_check
from utils.GLASSO.glasso import diff_test_check, cuda_diff_test_check

class TEST_CHECK():
    def __init__ (self,epsilon_tol,test_mode,timer,cuda):
        self.epsilon_tol = epsilon_tol
        self.test_mode = test_mode
        self.cuda = cuda
        self.timer = timer
        self.total_time = 0.0
        if self.cuda:
            import cupy as cp
            self.cp = cp

    def __call__(self, A, S, lam, A_inv = None):
        if self.timer:
            start_t = time.time()
        ret = self.check(A, S, lam, self.epsilon_tol, A_inv)
        if self.timer:
            end_t = time.time()
            self.total_time += end_t - start_t
        return ret

    def check(self, A, S, lam, epsilon, A_inv):
        if self.test_mode == "Gap":
            if self.cuda:
                return cuda_gap_test_check(self.cp, A, S, lam, epsilon, A_inv)
            else:
                return gap_test_check(A, S, lam, epsilon, A_inv)
        elif self.test_mode == "GapRel":
            if self.cuda:
                return cuda_gap_rel_test_check(self.cp, A, S, lam, epsilon, A_inv)
            else:
                return gap_rel_test_check(A, S, lam, epsilon, A_inv)
        elif self.test_mode == "Rel":
            if self.cuda:
                return cuda_rel_test_check(self.cp, A, S, lam, epsilon, A_inv)
            else:
                return rel_test_check(A, S, lam, epsilon, A_inv)
        elif self.test_mode == "Nmse":
            if self.cuda:
                return cuda_nmse_test_check(self.cp, A, epsilon, self.true_A)
            else:
                return nmse_test_check(A, epsilon, self.true_A)
        elif self.test_mode == "Diff":
            if self.cuda:
                return cuda_diff_test_check(self.cp, A, S, lam, epsilon, self.min_loss)
            else:
                return diff_test_check(A, S, lam, epsilon, self.min_loss)

    def get_total_time(self):
        return self.total_time

    def reset_total_time(self):
        self.total_time = 0

    def set_true_A(self, true_A):
        if self.test_mode == "Nmse":
            self.true_A = true_A
            if self.cuda:
                self.true_A = self.cp.array(self.true_A)

    def set_min_loss(self, min_loss):
        if self.test_mode == "Diff":
            self.min_loss = min_loss