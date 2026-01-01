import random


"""
Docstring for gemm

     N
  K  BB
     BB
M AA CC
  AA CC


"""
def gemm_nkm(A, B, C, M, N, K):
    for n in range(N):
        for k in range(K):
            for m in range(M):
                C[m][n] += A[m][k] * B[k][n]
    return C
