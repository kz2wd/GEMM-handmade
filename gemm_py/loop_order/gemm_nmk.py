import random


"""
Docstring for gemm

     N
  K  BB
     BB
M AA CC
  AA CC


"""
def gemm_nmk(A, B, C, M, N, K):
    for n in range(N):
        for m in range(M):
            for k in range(K):
                C[m][n] += A[m][k] * B[k][n]
    return C


