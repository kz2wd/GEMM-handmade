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


def random_gen(i, j):
    return random.random()


def prepare_continuous(M, N, K, init_a=random_gen, init_b=random_gen):
    A = [init_a(i % K, i // M) for i in range(M * K)]
    B = [init_b(i % N, i // K) for i in range(N * K)]
    C = [0 for _ in range(K * K)]
    return A, B, C



CHECK_SIZE = 256

def check_continuous(version):
    def determinist_location_gen(i, j):
        random.seed(i  * 31 + j * 23)
        return random.random()

    def get_mat():
        M = np.zeros((CHECK_SIZE, CHECK_SIZE))
        for (i, j), _ in np.ndenumerate(M):
            M[i, j] = determinist_location_gen(i, j)
        return M
    
    A = get_mat()
    B = get_mat()
    truth = A @ B

    A, B, C = version.prepare(CHECK_SIZE, CHECK_SIZE, CHECK_SIZE, determinist_location_gen, determinist_location_gen)
    C = version.run(A, B, C, CHECK_SIZE, CHECK_SIZE, CHECK_SIZE)
    
    
    return (C - truth.reshape(-1)).sum() 