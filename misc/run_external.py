import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")
import py_bigquic as pbq
import squic
import numpy as np
import time
import argparse
from datetime import timedelta
from utils.GLASSO.glasso import create_glasso_status

import os
import sys
from contextlib import contextmanager
from tempfile import TemporaryFile
def fileno(file_or_fd):
    fd = getattr(file_or_fd, 'fileno', lambda: file_or_fd)()
    if not isinstance(fd, int):
        raise ValueError("Expected a file (`.fileno()`) or a file descriptor")
    return fd

@contextmanager
def stdout_redirected(to=os.devnull, stdout=None):
    if stdout is None:
       stdout = sys.stdout

    stdout_fd = fileno(stdout)
    # copy stdout_fd before it is overwritten
    #NOTE: `copied` is inheritable on Windows when duplicating a standard stream
    with os.fdopen(os.dup(stdout_fd), 'wb') as copied:
        stdout.flush()  # flush library buffers that dup2 knows nothing about
        try:
            os.dup2(fileno(to), stdout_fd)  # $ exec >&to
        except ValueError:  # filename
            with open(to, 'wb') as to_file:
                os.dup2(to_file.fileno(), stdout_fd)  # $ exec > to
        try:
            yield stdout # allow code to be run with the redirected stdout
        finally:
            # restore stdout to its previous value
            #NOTE: dup2 makes stdout_fd inheritable unconditionally
            stdout.flush()
            os.dup2(copied.fileno(), stdout_fd)  # $ exec >&copied

def get_args():
    parser = argparse.ArgumentParser(description='External algorithm run')
    parser.add_argument(
        choices=['squic', 'bigquic'], default=None, dest='algorithm', help='External algorithm name.')
    parser.add_argument(
        '-tbs', '--tbs', type=int, required=True, default=5, dest='tbs', help='Test batch size.')
    parser.add_argument(
        '-db', '--db', type=str, required=True, default=None, dest='db',
        help='Pre generated database directory.')
    parser.add_argument(
        '-l', '--lam', required=True, type=float, default=0.1, dest='lam',
        help='The regularization paramater.')
    parser.add_argument(
        '-eps', '--eps', required=True, type=float, default=0.1, dest='eps',
        help='Tolerance.')
    parser.add_argument(
        '-T', '--T', required=True, type=int, default=15, dest='T',
        help="Number of iterations.")
    parser.add_argument(
        '-v', '--v', required=False, action='store_true', default=False, dest='v',
        help="Verbose.")
    parser.add_argument(
        '-vv', '--vv', required=False, action='store_true', default=False, dest='vv',
        help="Very verbose.")
    parser.add_argument(
        '-vvv', '--vvv', required=False, action='store_true', default=False, dest='vvv',
        help="Very very verbose.")
    return parser.parse_args()

def test_status(test_array, i):
    return "gap={gap:.6f} | loss={loss:.6f} | nnz={nnz:.6f} | grad={grad:.6f} | cond={cond:.6f} |  iter={iter:4.2f} | elpased={elpased} | call={call}".format(gap=test_array[i]['gap'], loss=test_array[i]['loss'], nnz=test_array[i]['nnz'], grad=test_array[i]['grad'], cond=test_array[i]['cond'], iter=test_array[i]['iter'], elpased=str(timedelta(seconds=test_array[i]['time'])), call=str(timedelta(seconds=test_array[i]['call'])))

def main():
    # parse configuration
    args = get_args()
    if args.vvv:
        verbose = 2
    elif args.vv:
        verbose = 1
    elif args.v:
        verbose = 0
    else:
        verbose = -1
    db = args.db
    #lam = np.float32(args.lam)
    lam = args.lam
    tbs = args.tbs
    #eps = np.float32(args.eps)
    eps = args.eps
    T = args.T
    algo = args.algorithm

    #ys = np.load(db+'/ys.npy').astype('float32')
    S = np.load(db+'/Ss.npy')
    ys = np.load(db+'/ys.npy').astype('float64')

    # start timer
    test_dict = {}
    test_dict['time'] = 0
    test_dict['call'] = 0
    test_dict['iter'] = np.float32(0)
    test_dict['gap'] = np.float32(0)
    test_dict['loss'] = np.float32(0)
    test_dict['nnz'] = np.float32(0)
    test_dict['grad'] = np.float32(0)
    test_dict['cond'] = np.float32(0)
    test_array = [test_dict.copy() for _ in range(tbs+1)]

    start = time.time()
    # start of task
    for i in range(tbs):
        if verbose < 0:
            ## Warmup
            f = TemporaryFile(mode='w+', encoding=sys.stdout.encoding)
            with stdout_redirected(f):
                if algo == 'squic': [_,_,_,_,_,_] = squic.run(ys[i], lam, 2, eps, 0)
                elif algo == 'bigquic': _ = pbq.bigquic(ys[i], lam, 2, eps, 0)
            f = TemporaryFile(mode='w+', encoding=sys.stdout.encoding)
            with stdout_redirected(f):
                start_call = time.time()
                if algo == 'squic': [A,_,_,_,_,_] = squic.run(ys[i], lam, T, eps, 0)
                elif algo == 'bigquic': A = pbq.bigquic(ys[i], lam, T, eps, 0)
                end_call = time.time()
                elapsed_call = end_call - start_call
            f.seek(0)
            s = f.read()
            iter=int((s.split("GLASSO_ITER"))[1].split("GLASSO_ITER")[0])
            elapsed=float((s.split("GLASSO_TIME"))[1].split("GLASSO_TIME")[0])
        else:
            start_call = time.time()
            if algo == 'squic':
                [A,_,list_times,list_objs,_,_] = squic.run(ys[i], lam, T, eps, verbose)
                iter = len(list_objs)
                elapsed = float(list_times[0])
            elif algo == 'bigquic':
                A = pbq.bigquic(ys[i], lam, T, eps, verbose)
                iter = np.nan # Can't know the number of iterations
                elapsed = 0 # Can't know the time it took
            end_call = time.time()
            elapsed_call = end_call - start_call
        if algo == 'squic': A = np.array(A.todense()) # To array for SQUIC only
        gap, loss, nnz, grad, _, cond, _ = create_glasso_status(S[i], lam)(A, 0)
        test_array[0]['time'] += elapsed
        test_array[0]['call'] += elapsed_call
        test_array[0]['iter'] += iter
        test_array[0]['gap'] += gap
        test_array[0]['loss'] += loss
        test_array[0]['nnz'] += nnz
        test_array[0]['grad'] += grad
        test_array[0]['cond'] += cond

        test_array[i+1]['time'] = elapsed
        test_array[i+1]['call'] = elapsed_call
        test_array[i+1]['iter'] = iter
        test_array[i+1]['gap'] = gap
        test_array[i+1]['loss'] = loss
        test_array[i+1]['nnz'] = nnz
        test_array[i+1]['grad'] = grad
        test_array[i+1]['cond'] = cond

    test_array[0]['time'] /= tbs
    test_array[0]['call'] /= tbs
    test_array[0]['iter'] /= tbs
    test_array[0]['gap'] /= tbs
    test_array[0]['loss'] /= tbs
    test_array[0]['nnz'] /= tbs
    test_array[0]['grad'] /= tbs
    test_array[0]['cond'] /= tbs
    # end of task

    print("Average    | {test_status}"\
      .format(test_status=test_status(test_array, 0)))

    for t in range(tbs):
        print("Index={t:4} | {test_status}"\
          .format(t=t+1,test_status=test_status(test_array, t+1)))

    # end timer
    end = time.time()
    elapsed = end - start
    print("Elapsed time = " + str(timedelta(seconds=elapsed)))


if __name__ == "__main__":
    main()
