import numpy as np


def numpy_naive(A, B, C, M, N, K):
    for k in range(K):
        for m in range(M):
            for n in range(N):
                C[m, n] = A[m, k] * B[k, n]

    return C
