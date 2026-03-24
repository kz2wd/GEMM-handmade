#include <stdio.h>
#include <stdlib.h>

#include "cgemm_kernel.h"
#include "cgemm_pack.h"
#include "cgemm_args.h"

#define prefetch_D 64
#define ALIGNEMENT 64
#define U 8
#define V 4
#define W 128

int main(int argc, char** argv){

    size_t K = 4096;

    double* A = (double *) aligned_alloc(64, sizeof(double) * K * K);
    double* B = (double *) aligned_alloc(64, sizeof(double) * K * K);
    double* C = (double *) aligned_alloc(64, sizeof(double) * K * K);

    init_mat(A, K);
    init_mat(B, K);
    memset(C, 0, sizeof(double) * K * K);

    // kernel_compute_intern(A, B, C, K);
    pack_compute_intern(A, K, B, K, C, K, K);

    return EXIT_SUCCESS;
}
