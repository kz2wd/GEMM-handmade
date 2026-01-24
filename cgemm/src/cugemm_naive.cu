#define PY_SSIZE_T_CLEAN
#include <Python.h>


#include "cugemm_naive.h"

#include "cgemm_args.h"

#include <stdlib.h>
#include <stdio.h>
#include <time.h>

// curated value, 16 or 32
#define THREAD_PER_BLOCK 16

__global__ void naive_gemm(double *A, double* B, double* C, int K) {
    // Embarassingly parallel condition
    // each thread compute one cell of C
    
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i > K || j > K) return;

    for (int k = 0; k < K; ++k) {
        at(C, i, j) += at(A, i, k) * at(B, k, j);
    }
    
}

PyObject* cu_naive_compute(PyObject* self, PyObject* args) {
    
    PyGEMMArgs* gemm_args;
    
    if (!PyArg_ParseTuple(args, "O!", &PyGEMMArgsType, &gemm_args)) return NULL;
    
    double* A = gemm_args->A;
    double* B = gemm_args->B;
    double* C = gemm_args->C;
    size_t K = gemm_args->K;

    double* d_A;
    double* d_B;
    double* d_C;

    size_t matrix_data_size = sizeof(double) * K * K;

    cudaMalloc((void**) &d_A, matrix_data_size);
    cudaMalloc((void**) &d_B, matrix_data_size);
    cudaMalloc((void**) &d_C, matrix_data_size);
    
    cudaMemcpy((void*) d_A, (void*) A, matrix_data_size, cudaMemcpyHostToDevice);
    cudaMemcpy((void*) d_B, (void*) B, matrix_data_size, cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();

    dim3 threadsPerBlock(THREAD_PER_BLOCK, THREAD_PER_BLOCK);
    dim3 blocks(K / THREAD_PER_BLOCK, K / THREAD_PER_BLOCK);
    naive_gemm<<<blocks, threadsPerBlock>>>(d_A, d_B, d_C, K);

    cudaDeviceSynchronize();

    cudaMemcpy((void*) C, (void*) d_C, matrix_data_size, cudaMemcpyDeviceToHost);

    cudaFree((void*) d_A);
    cudaFree((void*) d_B);
    cudaFree((void*) d_C);

    Py_INCREF(Py_None);
    return Py_None;
}