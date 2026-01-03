import numpy as np
import random
from utils import *


"""
Docstring for gemm

     N
  K  BB
     BB
M AA CC
  AA CC
"""
def gemm_naive(A, B, C, M, N, K):
    for m in range(M):
        for n in range(N):
            for k in range(K):
                C[m][n] += A[m][k] * B[k][n]
    return C




def prepare_naive(M, N, K, init_a=random_gen, init_b=random_gen):
    A = [[init_a(i, j) for j in range(K)] for i in range(M)]
    B = [[init_b(i, j) for j in range(N)] for i in range(K)]
    C = [[0 for _ in range(N)] for _ in range(K)]
    return A, B, C


CHECK_SIZE = 256

def check_naive(version):
    A = get_mat(determinist_location_gen, CHECK_SIZE, CHECK_SIZE)
    B = get_mat(determinist_location_gen, CHECK_SIZE, CHECK_SIZE)
    truth = A @ B

    A, B, C = version.prepare(CHECK_SIZE, CHECK_SIZE, CHECK_SIZE, determinist_location_gen, determinist_location_gen)
    C = version.run(A, B, C, CHECK_SIZE, CHECK_SIZE, CHECK_SIZE)
    
    
    return (C - truth).sum() 