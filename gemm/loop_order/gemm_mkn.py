import random


"""
Docstring for gemm

     N
  K  BB
     BB
M AA CC
  AA CC


"""
def gemm_mkn(A, B, C, M, N, K):
    for m in range(M):
        for k in range(K):
            for n in range(N):
                C[m][n] += A[m][k] * B[k][n]
    return C
