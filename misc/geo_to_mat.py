import argparse
import GEOparse as geo
import numpy as np
import pandas as pd
import os
import time
from datetime import timedelta

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

parser = argparse.ArgumentParser(description='GEO SOFT to npy file.')
parser.add_argument('path', metavar='PATH', type=str, nargs=1,
                    help='path to GEO SOFT file.')
parser.add_argument('col', metavar='COL', type=str, nargs=1,
                    help='name of the value\'s column.')
parser.add_argument( '-nr', '--normal', action='store_true', required=False,
                     default=False, dest='normal', help='normalize.')
parser.add_argument( '-cuda', '--cuda', action='store_true', required=False,
                     default=False, dest='cuda', help='use cuda.')
args = parser.parse_args()

start = time.time()

path = args.path[0]
col = args.col[0]
cuda = args.cuda
normal = args.normal

if cuda:
    import cupy as cp

g=geo.get_GEO(filepath=path)
name=g.metadata['geo_accession'][0]
df = pd.DataFrame()
for key in g.gsms:
    #df[key] = g.gsms[key].table.iloc[:,1]
    df[key] = g.gsms[key].table[col]
print(df)
if df.isnull().values.any():
    print("GEO data includes NaNs")
    df = df.dropna(axis=1)
    print(df)
if df.shape[1] == 0:
    print("No samples in GEO data")
    exit()
ys = df.to_numpy()
n = ys.shape[0]
samples = ys.shape[1]

print("N: {n} | samples: {samples}".format(n=n,samples=samples))

ss_list = []
sigs_list = [np.eye(n, dtype='float32')]
if not cuda:
    if normal:
        mean = compute_mean(ys)
        std = compute_std(ys)
        ys = normlize_data(ys, mean, std)
    ss_list += [cal_S(ys).astype('float32')]
else:
    ys = cp.asarray(ys)
    if normal:
        mean = cuda_compute_mean(cp, ys)
        std = cuda_compute_std(cp, ys)
        ys = cuda_normlize_data(cp, ys, mean, std)
    ss_list += [cp.asnumpy(cuda_cal_S(cp, ys)).astype('float32')]

db_path = 'db/' + name + '_N{n}_samples{samples}'.format(n=n,samples=samples)

if not os.path.exists(db_path):
    os.mkdir(db_path)

print("Save Sigs! Path:")
print(db_path+"/Sigs.npy")
np.save(db_path+"/Sigs.npy",np.array(sigs_list))

print("Save Ss! Path:")
print(db_path+"/Ss.npy")
np.save(db_path+"/Ss.npy",np.array(ss_list))

print("Save ys! Path:")
print(db_path+"/ys.npy")
np.save(db_path+"/ys.npy",np.array([ys]))

end = time.time()
elapsed = end - start
print("Elapsed time = " + str(timedelta(seconds=elapsed)))
