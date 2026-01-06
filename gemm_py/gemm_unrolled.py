
"""
Docstring for gemm

     N
  K  BB
     BB
M AA CC
  AA CC


"""
def gemm_unrolled2(A, B, C, M, N, K):
    for k in range(K):
        for m in range(M):
            for n in range(0, N // 2, 2):
                C[m][n + 0] += A[m][k] * B[k][n + 0]
                C[m][n + 1] += A[m][k] * B[k][n + 1]
    return C


def gemm_unrolled4(A, B, C, M, N, K):
    for k in range(K):
        for m in range(M):
            for n in range(0, N // 4, 4):
                C[m][n + 0] += A[m][k] * B[k][n + 0]
                C[m][n + 1] += A[m][k] * B[k][n + 1]
                C[m][n + 2] += A[m][k] * B[k][n + 2]
                C[m][n + 3] += A[m][k] * B[k][n + 3]
    return C


def gemm_unrolled8(A, B, C, M, N, K):
    for k in range(K):
        for m in range(M):
            for n in range(0, N // 8, 8):
                C[m][n + 0] += A[m][k] * B[k][n + 0]
                C[m][n + 1] += A[m][k] * B[k][n + 1]
                C[m][n + 2] += A[m][k] * B[k][n + 2]
                C[m][n + 3] += A[m][k] * B[k][n + 3]
                C[m][n + 4] += A[m][k] * B[k][n + 4]
                C[m][n + 5] += A[m][k] * B[k][n + 5]
                C[m][n + 6] += A[m][k] * B[k][n + 6]
                C[m][n + 7] += A[m][k] * B[k][n + 7]
    return C


def gemm_unrolled16(A, B, C, M, N, K):
    for k in range(K):
        for m in range(M):
            for n in range(0, N // 16, 16):
                C[m][n + 0] += A[m][k] * B[k][n + 0]
                C[m][n + 1] += A[m][k] * B[k][n + 1]
                C[m][n + 2] += A[m][k] * B[k][n + 2]
                C[m][n + 3] += A[m][k] * B[k][n + 3]
                C[m][n + 4] += A[m][k] * B[k][n + 4]
                C[m][n + 5] += A[m][k] * B[k][n + 5]
                C[m][n + 6] += A[m][k] * B[k][n + 6]
                C[m][n + 7] += A[m][k] * B[k][n + 7]
                C[m][n +  8] += A[m][k] * B[k][n +  8]
                C[m][n +  9] += A[m][k] * B[k][n +  9]
                C[m][n + 10] += A[m][k] * B[k][n + 10]
                C[m][n + 11] += A[m][k] * B[k][n + 11]
                C[m][n + 12] += A[m][k] * B[k][n + 12]
                C[m][n + 13] += A[m][k] * B[k][n + 13]
                C[m][n + 14] += A[m][k] * B[k][n + 14]
                C[m][n + 15] += A[m][k] * B[k][n + 15]
    return C


def gemm_unrolled32(A, B, C, M, N, K):
    for k in range(K):
        for m in range(M):
            for n in range(0, N // 32, 32):
                C[m][n + 0] += A[m][k] * B[k][n + 0]
                C[m][n + 1] += A[m][k] * B[k][n + 1]
                C[m][n + 2] += A[m][k] * B[k][n + 2]
                C[m][n + 3] += A[m][k] * B[k][n + 3]
                C[m][n + 4] += A[m][k] * B[k][n + 4]
                C[m][n + 5] += A[m][k] * B[k][n + 5]
                C[m][n + 6] += A[m][k] * B[k][n + 6]
                C[m][n + 7] += A[m][k] * B[k][n + 7]
                C[m][n +  8] += A[m][k] * B[k][n +  8]
                C[m][n +  9] += A[m][k] * B[k][n +  9]
                C[m][n + 10] += A[m][k] * B[k][n + 10]
                C[m][n + 11] += A[m][k] * B[k][n + 11]
                C[m][n + 12] += A[m][k] * B[k][n + 12]
                C[m][n + 13] += A[m][k] * B[k][n + 13]
                C[m][n + 14] += A[m][k] * B[k][n + 14]
                C[m][n + 15] += A[m][k] * B[k][n + 15]
                C[m][n + 16] += A[m][k] * B[k][n + 16]
                C[m][n + 17] += A[m][k] * B[k][n + 17]
                C[m][n + 18] += A[m][k] * B[k][n + 18]
                C[m][n + 19] += A[m][k] * B[k][n + 19]
                C[m][n + 20] += A[m][k] * B[k][n + 20]
                C[m][n + 21] += A[m][k] * B[k][n + 21]
                C[m][n + 22] += A[m][k] * B[k][n + 22]
                C[m][n + 23] += A[m][k] * B[k][n + 23]
                C[m][n + 24] += A[m][k] * B[k][n + 24]
                C[m][n + 25] += A[m][k] * B[k][n + 25]
                C[m][n + 26] += A[m][k] * B[k][n + 26]
                C[m][n + 27] += A[m][k] * B[k][n + 27]
                C[m][n + 28] += A[m][k] * B[k][n + 28]
                C[m][n + 29] += A[m][k] * B[k][n + 29]
                C[m][n + 30] += A[m][k] * B[k][n + 30]
                C[m][n + 31] += A[m][k] * B[k][n + 31]
    return C