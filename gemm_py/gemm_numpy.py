import numpy as np
from utils import *

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


def prepare_numpy(M, N, K, init_a=random_gen, init_b=random_gen):
    A = get_mat(init_a, M, K)
    B = get_mat(init_b, K, N)
    C = np.zeros((K, K))
    return A, B, C