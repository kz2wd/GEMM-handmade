import random


"""
Docstring for gemm

     N
  K  BB
     BB
M AA CC
  AA CC


"""
def gemm_kmn(A, B, C, M, N, K):
    for k in range(K):
        for m in range(M):
            for n in range(N):
                C[m][n] += A[m][k] * B[k][n]
    return C
