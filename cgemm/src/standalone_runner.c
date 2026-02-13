#include <stdio.h>
#include <stdlib.h>

#include "cgemm_kernel.h"
#include "cgemm_args.h"

int main(int argc, char** argv){

    size_t K = 2028;

    double* A = (double *) aligned_alloc(64, sizeof(double) * K * K);
    double* B = (double *) aligned_alloc(64, sizeof(double) * K * K);
    double* C = (double *) aligned_alloc(64, sizeof(double) * K * K);

    init_mat(A, K);
    init_mat(B, K);
    memset(C, 0, sizeof(double) * K * K);

    kernel_compute_intern(A, B, C, K);

    return EXIT_SUCCESS;
}
