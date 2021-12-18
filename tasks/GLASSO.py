import numpy as np
from utils.GLASSO.generate import GraphicalGenerator
from utils.GLASSO.glasso import create_glasso_status, cuda_create_glasso_status
from utils.GLASSO.TEST_CHECK import TEST_CHECK
from tasks.base import base_problem

class GLASSO(base_problem):
    def __init__ (self,algo,N,sig,min_samples,max_samples,type,type_param,id_add,normal,lam,test_mode,epsilon_tol,gista_T,cuda):
        super().__init__()
        self.algo = algo
        self.N = N
        self.sig = sig
        self.min_samples = min_samples
        self.max_samples = max_samples
        self.type = type
        self.type_param = type_param
        self.id_add = id_add
        self.normal = normal
        self.lam = lam
        self.epsilon_tol = epsilon_tol
        self.test_mode = test_mode

        self.cuda = cuda
        if self.cuda:
            import cupy as cp
            self.cp = cp

        self.gista_T = gista_T
        if self.gista_T is not None and self.gista_T > 0:
            if self.cuda:
                from algos.GLASSO.cuda_GISTA import cuda_GISTA as GISTA
            else:
                from algos.GLASSO.GISTA import GISTA
            self.gista = GISTA(T=self.gista_T, lam=self.lam, N=self.N, ls_iter=15, step_lim=1e-5)
        else:
            self.gista = None


    def name(self):
        if self.sig is not None:
            _const_sig = "_ConstSig"
        else:
            _const_sig = ""
        if self.normal:
            _normal = "_Normal"
        else:
            _normal = ""
        return 'GLASSO_lam{lam}_{algo_save_name}_{type}{type_param}_Iadd{id_add}_samples{min_samples}-{max_samples}{_const_sig}{_normal}'\
                .format(lam=self.lam,algo_save_name=self.algo.save_name,type=self.type,type_param=self.type_param,id_add=self.id_add,
                        min_samples=self.min_samples,max_samples=self.max_samples,
                        _const_sig=_const_sig,_normal=_normal)

    def init_full(self,seed,tbs,db,hist):
        self.hist = hist
        if self.sig is None:
            sig = None
        else:
            sig = np.load(self.sig)

        if db is None:
            generator = GraphicalGenerator(N=self.N, type=self.type, type_param=self.type_param, min_samples=self.min_samples,
                                         max_samples=self.max_samples,
                                         normal=self.normal, batch=tbs, constant=False, id_addition=self.id_add,
                                         sig=sig, cuda=self.cuda)
            self.Sigs, self.Ss = generator()
        else:
            self.Sigs = np.load(db + "/Sigs.npy").astype('float32')
            self.Ss = np.load(db + "/Ss.npy").astype('float32')

        self.As_hist = None
        if hist:
            self.As_hist = []

        if self.gista is not None:
            self.gista_As = [self.gista(S) for S in self.Ss]
        else:
            self.gista_As = [None for _ in self.Ss]

        self.gista_nmse = np.zeros(self.algo.T + 1, dtype='float32')
        self.loss = np.zeros(self.algo.T + 1, dtype='float32')
        self.nnz = np.zeros(self.algo.T + 1, dtype='float32')
        self.grad = np.zeros(self.algo.T + 1, dtype='float32')
        self.step = np.zeros(self.algo.T + 1, dtype='float32')
        self.cond = np.zeros(self.algo.T + 1, dtype='float32')
        self.gap = np.zeros(self.algo.T + 1, dtype='float32')

    def full(self):
        for (S, gista_A) in zip(self.Ss, self.gista_As):
            if not self.cuda:
                glasso_status = create_glasso_status(S, self.lam, gista_A)
            else:
                glasso_status = cuda_create_glasso_status(self.cp, S, self.lam, gista_A)
            if self.hist:
                _, status, As = self.algo.compute_full(S, glasso_status)
                self.As_hist.append(As)
            else:
                _, status = self.algo.compute_status(S, glasso_status)
            gap_A, loss_A, nnz_A, grad_A, step_A, cond_A, gista_nmse_A = list(zip(*status))
            if self.cuda:
                gap_A = self.cp.asnumpy(self.cp.array(gap_A))
                loss_A = self.cp.asnumpy(self.cp.array(loss_A))
                nnz_A = self.cp.asnumpy(self.cp.array(nnz_A))
                grad_A = self.cp.asnumpy(self.cp.array(grad_A))
                step_A = self.cp.asnumpy(self.cp.array(step_A))
                cond_A = self.cp.asnumpy(self.cp.array(cond_A))
                gista_nmse_A = self.cp.asnumpy(self.cp.array(gista_nmse_A))

            self.gap += np.array(gap_A)
            self.loss += np.array(loss_A)
            self.nnz += np.array(nnz_A)
            self.grad += np.array(grad_A)
            self.step += np.array(step_A)
            self.cond += np.array(cond_A)
            self.gista_nmse += np.array(gista_nmse_A)

        self.gap /= len(self.Ss)
        self.loss /= len(self.Ss)
        self.nnz /= len(self.Ss)
        self.grad /= len(self.Ss)
        self.step /= len(self.Ss)
        self.cond /= len(self.Ss)
        self.gista_nmse /= len(self.Ss)

    def full_status(self, t):
        if self.gista is None:
            return "gap={gap:.6f} | loss={loss:.6f} | nnz={nnz:.6f} | grad={grad:.6f} | step={step:.6f} | cond={cond:.6f}"\
                  .format(gap=self.gap[t], loss=self.loss[t], nnz=self.nnz[t],grad=self.grad[t],
                          step=self.step[t], cond=self.cond[t])
        else:
            return "gap={gap:.6f} | loss={loss:.6f} | gista_nmse={gista_nmse:.6f} | nnz={nnz:.6f} | grad={grad:.6f} | step={step:.6f} | cond={cond:.6f}" \
                .format(gap=self.gap[t], loss=self.loss[t], gista_nmse=self.gista_nmse[t], nnz=self.nnz[t], grad=self.grad[t],
                        step=self.step[t], cond=self.cond[t])

    def full_result(self):
        kwargs = {}
        kwargs['gap'] = self.gap
        kwargs['loss'] = self.loss
        kwargs['nnz'] = self.nnz
        kwargs['grad'] = self.grad
        kwargs['step'] = self.step
        kwargs['cond'] = self.cond
        kwargs['gista_nmse'] = self.gista_nmse

        return kwargs

    def full_hist(self):
        return self.As_hist

    def init_test(self,seed,tbs,db,hist,timer):
        self.timer = timer
        self.test_check_f = TEST_CHECK(self.epsilon_tol,self.test_mode,self.timer,self.cuda)
        self.hist = hist
        if self.sig is None:
            sig = None
        else:
            sig = np.load(self.sig)

        if db is None:
            generator = GraphicalGenerator(N=self.N, type=self.type, type_param=self.type_param, min_samples=self.min_samples,
                                         max_samples=self.max_samples,
                                         normal=self.normal, batch=tbs, constant=False, id_addition=self.id_add,
                                         sig=sig, cuda=self.cuda)
            self.Sigs, self.Ss = generator()
        else:
            self.Sigs = np.load(db + "/Sigs.npy").astype('float32')
            self.Ss = np.load(db + "/Ss.npy").astype('float32')

        self.As_hist = None
        if hist:
            self.As_hist = []

        if self.gista is not None:
            self.gista_As = [self.gista(S) for S in self.Ss]
        else:
            self.gista_As = [None for _ in self.Ss]

        test_dict = {}
        test_dict['time'] = 0
        test_dict['iter'] = np.float32(0)
        test_dict['gap'] = np.float32(0)
        test_dict['loss'] = np.float32(0)
        test_dict['nnz'] = np.float32(0)
        test_dict['grad'] = np.float32(0)
        test_dict['cond'] = np.float32(0)
        test_dict['gista_nmse'] = np.float32(0)

        self.test_array = [test_dict.copy() for _ in range(tbs+1)]

    def test(self):
        for i in range(len(self.Ss)):
            S = self.Ss[i]
            Sig = self.Sigs[i]
            gista_A = self.gista_As[i]
            if not self.cuda:
                glasso_status = create_glasso_status(S, self.lam, gista_A)
            else:
                glasso_status = cuda_create_glasso_status(self.cp, S, self.lam, gista_A)
            import time
            self.algo.compute_warmup(S)
            self.test_check_f.set_true_A(Sig)
            if not self.cuda:
                self.test_check_f.set_min_loss(glasso_status(Sig, 0.0)[1])
            else:
                self.test_check_f.set_min_loss(glasso_status(self.cp.array(Sig), 0.0)[1])
            self.test_check_f.reset_total_time()
            start_t = time.time()
            A, iter_A = self.algo.compute_test(S, self.test_check_f)
            test_t = time.time()
            elapsed = test_t - start_t - self.test_check_f.get_total_time()
            if self.hist:
                self.As_hist.append(A)

            gap_A, loss_A, nnz_A, grad_A, step_A, cond_A, gista_nmse_A = glasso_status(A, 0.0)
            if self.cuda:
                gap_A = np.float32(self.cp.asnumpy(gap_A))
                loss_A = np.float32(self.cp.asnumpy(loss_A))
                nnz_A = np.float32(self.cp.asnumpy(nnz_A))
                grad_A = np.float32(self.cp.asnumpy(grad_A))
                cond_A = np.float32(self.cp.asnumpy(cond_A))
                gista_nmse_A = np.float32(self.cp.asnumpy(gista_nmse_A))


            self.test_array[0]['time'] += elapsed
            self.test_array[0]['iter'] += iter_A
            self.test_array[0]['gap'] += gap_A
            self.test_array[0]['loss'] += loss_A
            self.test_array[0]['nnz'] += nnz_A
            self.test_array[0]['grad'] += grad_A
            self.test_array[0]['cond'] += cond_A
            self.test_array[0]['gista_nmse'] += gista_nmse_A

            self.test_array[i+1]['time'] = elapsed
            self.test_array[i+1]['iter'] += iter_A
            self.test_array[i+1]['gap'] = gap_A
            self.test_array[i+1]['loss'] = loss_A
            self.test_array[i+1]['nnz'] = nnz_A
            self.test_array[i+1]['grad'] = grad_A
            self.test_array[i+1]['cond'] = cond_A
            self.test_array[i+1]['gista_nmse'] = gista_nmse_A

        self.test_array[0]['time'] /= len(self.Ss)
        self.test_array[0]['iter'] /= len(self.Ss)
        self.test_array[0]['gap'] /= len(self.Ss)
        self.test_array[0]['loss'] /= len(self.Ss)
        self.test_array[0]['nnz'] /= len(self.Ss)
        self.test_array[0]['grad'] /= len(self.Ss)
        self.test_array[0]['cond'] /= len(self.Ss)
        self.test_array[0]['gista_nmse'] /= len(self.Ss)

    def test_status(self, i):
        from datetime import timedelta
        if self.gista is None:
            return "gap={gap:.6f} | loss={loss:.6f} | nnz={nnz:.6f} | grad={grad:.6f} | cond={cond:.6f} |  iter={iter:4.2f} | elpased={elpased}"\
                  .format(gap=self.test_array[i]['gap'], loss=self.test_array[i]['loss'], nnz=self.test_array[i]['nnz'],grad=self.test_array[i]['grad'],
                          cond=self.test_array[i]['cond'], iter=self.test_array[i]['iter'], elpased=str(timedelta(seconds=self.test_array[i]['time'])))
        else:
            return "gap={gap:.6f} | loss={loss:.6f} | gista_nmse={gista_nmse:.6f}  | nnz={nnz:.6f} | grad={grad:.6f} | cond={cond:.6f} |  iter={iter:4.2f}  | elpased={elpased}" \
                .format(gap=self.test_array[i]['gap'], loss=self.test_array[i]['loss'], gista_nmse=self.test_array[i]['gista_nmse'], nnz=self.test_array[i]['nnz'],
                        grad=self.test_array[i]['grad'], cond=self.test_array[i]['cond'], iter=self.test_array[i]['iter'],
                        elpased=str(timedelta(seconds=self.test_array[i]['time'])))

    def test_result(self):
        kwargs = {}
        for i in range(len(self.test_array)):
            kwargs['{i}'.format(i=i)] = self.test_array[i]
        return kwargs

    def test_hist(self):
        return self.As_hist

    def init_generate(self, object, seed, tbs):
        if self.sig is None:
            sig = None
        else:
            sig = np.load(self.sig)

        self.generator = GraphicalGenerator(N=self.N, type=self.type, type_param=self.type_param, min_samples=self.min_samples,
                                         max_samples=self.max_samples,
                                         normal=self.normal, batch=tbs, constant=False, id_addition=self.id_add,
                                         sig=sig, cuda=self.cuda)

        if self.sig is not None:
            _const_sig = "_ConstSig"
        else:
            _const_sig = ""
        if self.normal:
            _normal = "_Normal"
        else:
            _normal = ""
        self.gen_name = 'N{N}_{type}{type_param}_Iadd{id_add}_samples{min_samples}-{max_samples}{_const_sig}{_normal}' \
            .format(N=self.N, type=self.type, type_param=self.type_param, id_add=self.id_add,
                    min_samples=self.min_samples, max_samples=self.max_samples,
                    _const_sig=_const_sig, _normal=_normal)

        self.nnz = np.zeros(tbs + 1, dtype='float32')

    def generate_name(self):
        return self.gen_name

    def generate(self):
        self.Sigs, self.Ss = self.generator()
        for i in range(len(self.Sigs)):
            self.nnz[i+1] = np.count_nonzero(self.Sigs[i])
            self.nnz[0] += self.nnz[i+1]
        self.nnz[0] /= len(self.Sigs)

    def generate_status(self, i):
        return "nnz={nnz:.6f} | nnz_precent={nnz_precent:.6f}" \
              .format(nnz=self.nnz[i], nnz_precent=self.nnz[i] / (self.N ** 2) * 100)

    def generate_save(self, path):
        np.save(path + "/Sigs.npy", np.array(self.Sigs))
        np.save(path + "/Ss.npy", np.array(self.Ss))

