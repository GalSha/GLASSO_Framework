import numpy as np
import scipy as sp
import scipy.sparse as spr
import scipy.spatial as spt

def numgrid(n):
    grid = np.zeros((n,n),dtype='int64')
    counter = 1
    for j in range(1, n - 1):
        for i in range(1,n-1):
            grid[i,j] = counter
            counter+=1
    return grid

def delsq(grid):
    n = grid.shape[0]
    m = grid.shape[1]
    size_n = np.max(grid)
    laplace_padded_grid = np.zeros((size_n+1,size_n+1))
    for i in range(1,n-1):
        for j in range(1,m-1):
            if grid[i,j] != 0:
                laplace_padded_grid[grid[i,j],grid[i,j]] = 4
                if grid[i-1,j] != 0:
                    laplace_padded_grid[grid[i,j],grid[i-1,j]] = -1
                if grid[i,j-1] != 0:
                    laplace_padded_grid[grid[i,j],grid[i,j-1]] = -1
                if grid[i+1,j] != 0:
                    laplace_padded_grid[grid[i,j],grid[i+1,j]] = -1
                if grid[i,j+1] != 0:
                    laplace_padded_grid[grid[i,j],grid[i,j+1]] = -1
    laplace_grid = np.zeros((size_n,size_n))
    laplace_grid[:,:] = laplace_padded_grid[1:,1:]
    return spr.coo_matrix(laplace_grid)

def get_rescale_matrix(A):
    return spr.coo_matrix(np.diag(1./ np.sqrt(A.diagonal(0))))

def generateRandomPlanarGraph(n):
    x = np.zeros(n)
    y = np.zeros(n)
    while np.any(x==0): x = np.random.rand(1,n)
    while np.any(y==0): y = np.random.rand(1,n)
    I = np.argsort(y)
    y = np.sort(y)
    x[0,:] = x[0,I]
    p = np.zeros((2,n))
    p[0,:]=x
    p[1,:]=y
    p = p.T
    TRI = spt.Delaunay(p).simplices
    D = np.ones((3*TRI.shape[0],3),dtype='int64')
    for i in range(TRI.shape[0]):
        D[3 * i    , 0:2] = TRI[i, [0,1]]
        D[3 * i + 1, 0:2] = TRI[i, [1,2]]
        D[3 * i + 2, 0:2] = TRI[i, [0,2]]
    A = spr.coo_matrix((D[:,2],(D[:,0],D[:,1])),(n,n),dtype='float64')
    A = A + A.T
    A[A>0] = 1.
    return A

def DistanceMat(AdjMat, max_distance):
    Routes = np.eye(AdjMat.shape[0], dtype='float32')
    distMat = np.zeros(AdjMat.shape, dtype='float32')
    for d in range(max_distance):
        Routes = Routes @ AdjMat
        Add = (d+1) * np.sign(Routes, dtype='float32')
        Add[distMat!=0] = 0
        distMat += Add
    np.fill_diagonal(distMat, 0)
    return spr.coo_matrix(distMat)

def DistanceLaplace(AdjMat, max_distance):
    distMat = DistanceMat(AdjMat, max_distance)
    distMat_sum = np.ravel(np.sum(distMat,axis=0))
    D = spr.coo_matrix(np.diag(distMat_sum))
    return D-distMat

def generateRandomPlanarDistanceLaplace(n, max_distance, id_addition=1e-16):
    Adj = generateRandomPlanarGraph(n)
    A = DistanceLaplace(Adj, max_distance) #+ id_addition*spr.eye(n)
    #if id_addition == 0:
    lmin = getLambdaMin(A.todense())
    delta = np.max((-1.2 * lmin, id_addition))
    A = A + delta * spr.eye(n)
    #rescaling = get_rescale_matrix(A)
    #A = rescaling@A@rescaling
    return A

def generateChain(n,id_addition=1e-16):
    e = np.ones(n)
    A = np.diag(e,0) + np.diag(-0.5*e[:-1],1) + np.diag(-0.5*e[:-1],-1) #+ id_addition*np.eye(n)
    #if id_addition == 0:
    lmin = getLambdaMin(A)
    delta = np.max((-1.2 * lmin, id_addition))
    A = A + delta * spr.eye(n)
    return spr.coo_matrix(A)

def getLambdaMin(A):
    return np.linalg.eigvalsh(A)[0]
    rho = np.linalg.norm(A,1)
    omega = 2/(3*rho)
    x = np.ones((A.shape[0],1))
    for _ in range(5):
        for _ in range(3):
            x = x- omega * (A.T @ x)
        x = x / np.sqrt(x.T@x)
    lambda_min = x.T @ A.T @ x
    if lambda_min < 0: lambda_min *= 1.2
    else: lambda_min *= 0.8
    return lambda_min

def generateRandom(n,density,id_addition=1e-16):
    A_rand = spr.random(n,n,np.sqrt(density*n/100)/n).toarray()
    A = np.zeros((n,n),dtype='float64')
    A[A_rand != 0] += -1
    A[A_rand > 0.5] += 2
    A = A.T @ A
    d = np.diag(A).copy()
    A -= np.diag(d)
    A[A>1] = 1
    A[A<-1] = -1
    #A += np.diag(d+id_addition)
    A += np.diag(d)
    lmin = getLambdaMin(A)
    delta = np.max((-1.2 * lmin, id_addition))
    A = A + delta * spr.eye(n)
    return spr.coo_matrix(A)

def generate_ys(Sigma_inv,samples=1):
    n = Sigma_inv.shape[0]
    L_inv = np.linalg.inv(np.linalg.cholesky(Sigma_inv))
    xs = np.random.normal(0, 1, [samples, n])
    #xs = np.random.multivariate_normal(np.zeros(n), np.eye(n), samples)
    ys = L_inv.T@xs.T
    return ys

def cuda_generate_ys(cp,Sigma_inv,samples=1):
    n = Sigma_inv.shape[0]
    L_inv = cp.linalg.inv(cp.linalg.cholesky(Sigma_inv))
    xs = cp.random.normal(0, 1, (samples, n))
    #xs = np.random.multivariate_normal(np.zeros(n), np.eye(n), samples)
    ys = L_inv.T@xs.T
    return ys

def cal_S(ys):
    n = ys.shape[0]
    samples = ys.shape[1]
    S = np.zeros((n,n),dtype='float64')
    for i in range(samples):
        y_i = ys[:,i].reshape(-1,1)
        S += y_i@y_i.T
    S /= samples
    return S

def cuda_cal_S(cp,ys):
    n = ys.shape[0]
    samples = ys.shape[1]
    S = cp.zeros((n,n),dtype='float64')
    for i in range(samples):
        y_i = ys[:,i].reshape(-1,1)
        S += y_i@y_i.T
    S /= samples
    return S

def compute_mean(ys):
    return np.mean(ys,axis=1)

def cuda_compute_mean(cp,ys):
    return cp.mean(ys,axis=1)

def compute_std(ys):
    return np.std(ys,axis=1)

def cuda_compute_std(cp,ys):
    return cp.std(ys,axis=1)

def normlize_data(ys,mean,std):
    mean_mat = np.repeat(mean.reshape(-1,1),axis=1,repeats=ys.shape[1])
    ys = ys - mean_mat
    std_mat = np.diag(1/std)
    ys = std_mat@ys
    return ys

def cuda_normlize_data(cp,ys,mean,std):
    mean_mat = cp.repeat(mean.reshape(-1,1),axis=1,repeats=ys.shape[1])
    ys = ys - mean_mat
    std_mat = cp.diag(1/std)
    ys = std_mat@ys
    return ys

def normlize_sig_inv(sig_inv, std):
    std_mat = np.diag(std)
    return std_mat@sig_inv@std_mat

def cuda_normlize_sig_inv(cp,sig_inv, std):
    std_mat = cp.diag(std)
    return std_mat@sig_inv@std_mat

class GraphicalGenerator():
    def __init__(self, N, type, type_param, min_samples, max_samples, normal, batch, constant, id_addition = 1, sig = None, cuda=False):
        self.N = N
        self.type = type
        self.type_param = type_param
        self.min_samples = min_samples
        self.max_samples = max_samples
        self.normal = normal
        self.batch = batch
        self.constant = constant
        self.sig = sig
        self.sig_exist = sig is not None
        self.id_addition = id_addition
        self.cuda = cuda
        if self.cuda:
            import cupy as cp
            self.cp = cp

        self.sigs_list = []
        self.ss_list = []
        self.ys_list = []
        if self.constant:
            for _ in range(self.batch):
                if self.sig_exist:
                    self.sigs_list += [self.sig.astype('float32')]
                elif self.type == "chain":
                    self.sigs_list += [generateChain(self.N, self.id_addition).toarray().astype('float32')]
                elif self.type == "planar":
                    self.sigs_list += [generateRandomPlanarDistanceLaplace(self.N, int(self.type_param), self.id_addition).toarray().astype('float32')]
                elif self.type == "random":
                    self.sigs_list += [generateRandom(self.N, self.type_param, self.id_addition).toarray().astype('float32')]
                samples = np.random.randint(low=self.min_samples, high=self.max_samples + 1)
                if not self.cuda:
                    ys = generate_ys(self.sigs_list[-1], samples)
                    if self.normal:
                        mean = compute_mean(ys)
                        std = compute_std(ys)
                        ys = normlize_data(ys, mean, std)
                        self.sigs_list[-1] = normlize_sig_inv(self.sigs_list[-1], std).astype('float32')
                    self.ss_list += [cal_S(ys).astype('float32')]
                    self.ys_list += [ys.astype('float32')]
                else:
                    cuda_sig = self.cp.asarray(self.sigs_list[-1])
                    ys = cuda_generate_ys(self.cp,cuda_sig, samples)
                    if self.normal:
                        mean = cuda_compute_mean(self.cp,ys)
                        std = cuda_compute_std(self.cp,ys)
                        ys = cuda_normlize_data(self.cp,ys, mean, std)
                        self.sigs_list[-1] = self.cp.asnumpy(cuda_normlize_sig_inv(self.cp, cuda_sig, std)).astype(
                            'float32')
                    self.ss_list += [self.cp.asnumpy(cuda_cal_S(self.cp, ys)).astype('float32')]
                    self.ys_list += [self.cp.asnumpy(ys).astype('float32')]

    def __call__(self):
        if self.constant:
            return self.sigs_list, self.ss_list

        self.sigs_list = []
        self.ss_list = []
        self.ys_list = []
        for _ in range(self.batch):
            if self.sig_exist:
                self.sigs_list += [self.sig.astype('float32')]
            elif self.type == "chain":
                self.sigs_list += [generateChain(self.N, self.id_addition).toarray().astype('float32')]
            elif self.type == "planar":
                self.sigs_list += [generateRandomPlanarDistanceLaplace(self.N, int(self.type_param), self.id_addition).toarray().astype('float32')]
            elif self.type == "random":
                self.sigs_list += [generateRandom(self.N, self.type_param, self.id_addition).toarray().astype('float32')]
            samples = np.random.randint(low=self.min_samples, high=self.max_samples + 1)
            if not self.cuda:
                ys = generate_ys(self.sigs_list[-1], samples)
                if self.normal:
                    mean = compute_mean(ys)
                    std = compute_std(ys)
                    ys = normlize_data(ys, mean, std)
                    self.sigs_list[-1] = normlize_sig_inv(self.sigs_list[-1], std).astype('float32')
                self.ss_list += [cal_S(ys).astype('float32')]
                self.ys_list += [ys.astype('float32')]
            else:
                cuda_sig = self.cp.asarray(self.sigs_list[-1])
                ys = cuda_generate_ys(self.cp, cuda_sig, samples)
                if self.normal:
                    mean = cuda_compute_mean(self.cp, ys)
                    std = cuda_compute_std(self.cp, ys)
                    ys = cuda_normlize_data(self.cp, ys, mean, std)
                    self.sigs_list[-1] = self.cp.asnumpy(cuda_normlize_sig_inv(self.cp, cuda_sig, std)).astype(
                        'float32')
                self.ss_list += [self.cp.asnumpy(cuda_cal_S(self.cp, ys)).astype('float32')]
                self.ys_list += [self.cp.asnumpy(ys).astype('float32')]

        return self.sigs_list, self.ss_list, self.ys_list
