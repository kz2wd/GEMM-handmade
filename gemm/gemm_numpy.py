import numpy as np


"""
Docstring for gemm

     N
  K  BB
     BB
M AA CC
  AA CC


"""
def gemm_numpy(A, B, C, M, N, K):
    C = A @ B
    return C


def prepare_numpy(M, N, K):
    A = np.random.rand(M, K)
    B = np.random.rand(K, N)
    C = np.zeros((K, K))
    return A, B, C