import random


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


def prepare_naive(M, N, K):
    A = [[random.random() for _ in range(K)] for _ in range(M)]
    B = [[random.random() for _ in range(N)] for _ in range(K)]
    C = [[0 for _ in range(N)] for _ in range(K)]
    return A, B, C

