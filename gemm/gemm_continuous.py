import random


"""
Docstring for gemm

     N
  K  BB
     BB
M AA CC
  AA CC


"""
def gemm_continuous(A, B, C, M, N, K):
    for k in range(K):
        for m in range(M):
            for n in range(N):
                C[m + n * M] += A[m + k * M] * B[k + n * K]
    return C

def prepare_continuous(M, N, K):
    A = [random.random() for _ in range(M * K)]
    B = [random.random() for _ in range(N * K)]
    C = [0 for _ in range(K * K)]
    return A, B, C

