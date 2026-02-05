import numpy as np
from time import perf_counter


from gemm_py.gemm_naive import gemm_naive, prepare_naive, check_naive
from gemm_py.gemm_numpy import gemm_numpy, prepare_numpy, prepare_numpy_float
from gemm_py.loop_order.gemm_kmn import gemm_kmn
from gemm_py.loop_order.gemm_knm import gemm_knm   
from gemm_py.loop_order.gemm_mkn import gemm_mkn
from gemm_py.loop_order.gemm_mnk import gemm_mnk
from gemm_py.loop_order.gemm_nkm import gemm_nkm
from gemm_py.loop_order.gemm_nmk import gemm_nmk
from gemm_py.gemm_continuous import gemm_continuous, prepare_continuous, check_continuous
from gemm_py.gemm_numpy_naive import numpy_naive
from gemm_py.gemm_unrolled import gemm_unrolled2, gemm_unrolled4, gemm_unrolled8, gemm_unrolled16, gemm_unrolled32

import sys
from pathlib import Path
sys.path.append("cgemm/build/src")
import cgemm

from collections.abc import Callable
from dataclasses import dataclass

@dataclass
class DataLayout:
    prepare: Callable
    check: Callable
    benchmark: Callable

@dataclass
class GEMM:
    name: str
    precision: str
    device: str
    language: str
    run: Callable
    layout: DataLayout


epsilon = 1e-6


def naive_benchmark(version: GEMM, S):
    A, B, C = version.layout.prepare(S, S, S)

    t1 = perf_counter()
    version.run(A, B, C, S, S, S)
    t2 = perf_counter()

    return t2 - t1


def cbenchmark(version: GEMM, K):
    cgemm_args = version.layout.prepare(K)

    t1 = perf_counter()
    version.run(cgemm_args)
    t2 = perf_counter()

    return t2 - t1

def check_numpy(_):
    # Numpy is used as ground truth
    return 0

def check_c(version: GEMM):
    epsilon = 1e-6
    K = 128
    cgemm_args = version.layout.prepare(K)
    version.run(cgemm_args)
    A = np.zeros((K, K))
    B = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            A[i, j] = cgemm.get_naive(cgemm_args, 0, i, j)
            B[i, j] = cgemm.get_naive(cgemm_args, 1, i, j)
    C = A @ B
    error = 0
    for i in range(K):
        for j in range(K):
            error += C[i, j] - cgemm.get_naive(cgemm_args, 2, i, j)
    return error


def check(version: GEMM):
    error = version.layout.check(version)
    if error > epsilon:
        raise RuntimeError(f'Version [{version.name}] failed with an error of {error:.5f}')
    else:
        print(f'Version [{version.name}] is performing correctly')


def benchmark(version: GEMM, K):
    return version.layout.benchmark(version, K)  




numpy_layout = DataLayout(prepare_numpy, check_numpy, naive_benchmark)
naive_layout = DataLayout(prepare_naive, check_naive, naive_benchmark)
continuous_layout = DataLayout(prepare_continuous, check_continuous, naive_benchmark)
cnaive_layout = DataLayout(cgemm.naive_prepare, check_c, cbenchmark)
numpy_FP32_layout = DataLayout(prepare_numpy_float, check_numpy, naive_benchmark)
caligned_layout = DataLayout(cgemm.aligned_memory_prepare, check_c, cbenchmark)


kernels = {
    'numpy': GEMM('numpy', 'FP64', 'CPU', 'python', gemm_numpy, numpy_layout),
    'numpy_naive': GEMM('numpy naive', 'FP64', 'CPU', 'python', numpy_naive, numpy_layout,),
    'naive_py': GEMM('python naive', 'FP64', 'CPU', 'python', gemm_naive, naive_layout),
    'loop_kmn': GEMM('loop kmn', 'FP64', 'CPU', 'python', gemm_kmn, naive_layout),
    'loop_knm': GEMM('loop knm', 'FP64', 'CPU', 'python', gemm_knm, naive_layout),
    'loop_mkn': GEMM('loop mkn', 'FP64', 'CPU', 'python', gemm_mkn, naive_layout),
    'loop_mnk': GEMM('loop mnk', 'FP64', 'CPU', 'python', gemm_mnk, naive_layout),
    'loop_nkm': GEMM('loop nkm', 'FP64', 'CPU', 'python', gemm_nkm, naive_layout),
    'loop_nmk': GEMM('loop nmk', 'FP64', 'CPU', 'python', gemm_nmk, naive_layout),
    'continuous_memory': GEMM('continuous memory', 'FP64', 'CPU', 'python', gemm_continuous, continuous_layout),
    'unrolled2_py': GEMM('unrolled2 py', 'FP64', 'CPU', 'python', gemm_unrolled2, naive_layout), 
    'unrolled4_py': GEMM('unrolled4 py', 'FP64', 'CPU', 'python', gemm_unrolled4, naive_layout),
    'unrolled8_py': GEMM('unrolled8 py', 'FP64', 'CPU', 'python', gemm_unrolled8, naive_layout),
    'unrolled16_py': GEMM('unrolled16 py', 'FP64', 'CPU', 'python', gemm_unrolled16, naive_layout), 
    'unrolled32_py': GEMM('unrolled32 py', 'FP64', 'CPU', 'python', gemm_unrolled32, naive_layout),
    'naive_c': GEMM("naive c", 'FP64', 'CPU', 'c', cgemm.naive_compute, cnaive_layout),
    'naive_acc_compute': GEMM("naive accumulator", 'FP64', 'CPU', 'c', cgemm.naive_acc_compute, cnaive_layout),
    'naive_cu': GEMM("naive cuda", 'FP64', 'GPU', 'cuda', cgemm.cu_naive_compute, cnaive_layout),
    'numpy_FP32': GEMM("numpy FP32", 'FP32', 'CPU', 'python', gemm_numpy, numpy_FP32_layout),
    'block_c': GEMM("blocked c", 'FP64', 'CPU', 'c', cgemm.block_compute, cnaive_layout),
    'kernel_c': GEMM("kernel c", 'FP64', 'CPU', 'c', cgemm.kernel_compute, caligned_layout),
}
