# GLASSO framework
Python implementation of a generic framework for the GLASSO problem - including data generation and algorithm testing.
  
## Requirements

- Python3.9.
- Numpy1.20.1.
- [Optional] Cupy9.0 – needed only if one wants to use GPU.

## Instructions

The main file is `main.py` which use various command line arguments and flags.
The first command line argument is the problem, currently the only supported problem is `glasso`.
The second command line argument is the task to run. It can be either:
1. `generate` - generate data. The next command line argument should specify the object to generate, currently the only supported problem is `data`.
2. `full` - run an algorithm with full logging of each iteration. This task should not be used for timing an algorithm. The next command line argument should specify the algorithm, currently the only supported algorithms are `pISTA`, `OBN`, `QUIC`, `GISTA`, `ALM` and `NL_fista`.
3. `test` - run an algorithm until an end criterion (timing test). The next command line argument should specify the algorithm, currently the only supported algorithms are `pISTA`, `OBN`, `QUIC`, `GISTA`, `ALM` and `NL_fista`.
Afterwards, one can use the various flags to configure the task.

For additional details, use the `-h` flag.

An example for data generation task:
```
python main.py gen data -N 10000 -nr --type random -id_add 0.1 -min_samples 300 -max_samples 300 -tbs 5 --type_param 0.5
```

The output of the data generation is kept in `db` folder.

An example for algorithm timing task:
```
python main.py glasso test pISTA -eps 0.01 -nr -N 10000 --type random --type_param 0.5 -l 0.2 -id_add 0.1 -max_samples 300 -min_samples 300 -tbs 5 -T 200 -cr Rel
```
The results of the algorithm iterations and logs are kept in `results` folder.
One can use the `-db` flag to use pre-defined matrix: `-db ./db/GLASSO/N10000_random0.5_Iadd0.1_samples300-300_Normal_id0`

## GPU and CUDA ##

If the Cupy library is installed, one can use the `-cuda` flag to run the task on the GPU - whether it is a data generation or an algorithm.

## MISC ##

One can use the `misc/geo_to_mat.py` script to turn a GEO SOFT file as in [Gene Expression Omnibus](http://www.ncbi.nlm.nih.gov/geo/) into a matrix.

## References
### Papers
* `QUIC` algorithm - [QUIC: Quadratic Approximation for Sparse Inverse Covariance Estimation](https://jmlr.org/papers/volume15/hsieh14a/hsieh14a.pdf) (Cho-Jui Hsieh, Mátyás A. Sustik, Inderjit S. Dhillon, Pradeep Ravikumar)  
* `GISTA` algorithm - [Iterative Thresholding Algorithm for Sparse Inverse Covariance Estimation](https://arxiv.org/pdf/1211.2532.pdf) (Dominique Guillot, Bala Rajaratnam, Benjamin T. Rolfs, Arian Maleki, Ian Wong)  
* `ALM` algorithm - [Sparse inverse covariance selection via alternating linearization methods](https://proceedings.neurips.cc/paper/2010/file/2723d092b63885e0d7c260cc007e8b9d-Paper.pdf) (Katya Scheinberg, Shiqian Ma, Donald Goldfarb.)  
* `OBN` and `NL_fista` algorithms - [Newton-like methods for sparse inverse covariance estimation](https://papers.nips.cc/paper/2012/file/b3967a0e938dc2a6340e258630febd5a-Paper.pdf) (Figen Oztoprak, Jorge Nocedal, Steven Rennie, Peder A. Olsen)  

### Implementations
  * [ALM authors implemeanations](https://www.math.ucdavis.edu/~sqma/ALM-SICS.html)
