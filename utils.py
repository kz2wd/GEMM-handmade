import numpy as np
import random

def get_mat(generation, I, J):
    M = np.zeros((I, J))
    for (i, j), _ in np.ndenumerate(M):
        M[i, j] = generation(i, j)
    return M

def determinist_location_gen(i, j):
        random.seed(i  * 31 + j * 23)
        return random.random()

def random_gen(i, j):
    return random.random()

