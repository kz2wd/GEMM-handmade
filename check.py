import numpy as np
import random


CHECK_SIZE = 256



def check(version):
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
    

    return (C - truth).sum() 
