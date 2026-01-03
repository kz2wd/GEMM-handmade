import numpy as np
import pandas as pd
from time import perf_counter
from tqdm import tqdm
from gemm_py.gemm_naive import gemm_naive, prepare_naive, check_naive
from gemm_py.gemm_numpy import gemm_numpy, prepare_numpy
from gemm_py.loop_order.gemm_kmn import gemm_kmn
from gemm_py.loop_order.gemm_knm import gemm_knm   
from gemm_py.loop_order.gemm_mkn import gemm_mkn
from gemm_py.loop_order.gemm_mnk import gemm_mnk
from gemm_py.loop_order.gemm_nkm import gemm_nkm
from gemm_py.loop_order.gemm_nmk import gemm_nmk
from gemm_py.gemm_continuous import gemm_continuous, prepare_continuous
from gemm_py.gemm_numpy_naive import numpy_naive
from dataclasses import dataclass
import sqlite3
from collections.abc import Callable

@dataclass
class GEMM:
    name: str
    run: Callable
    prepare: Callable


def benchmark(version: GEMM, M, N, K):
    A, B, C = version.prepare(M, N, K)

    t1 = perf_counter()
    version.run(A, B, C, M, N, K)
    t2 = perf_counter()

    return t2 - t1


epsilon = 1e-6
def check_versions():
    # version = GEMM('naive_py', gemm_naive, prepare_naive)
    version = GEMM('numpy_naive', numpy_naive, prepare_numpy)
    error = check_naive(version)
    if error > epsilon:
        print(f'Version [{version.name}] failed with an error of {error:.5f}')
    else:
        print(f'Version [{version.name}] performed correctly')


def main():
    conn = sqlite3.connect("benchmarks.db")


    # sizes = [128, 256, 512]
    sizes = [1024]
    # sizes = [2048, 4096, 8192]

    warmup = 1
    trials = 5
    # versions = [GEMM('naive_py', gemm_naive, prepare_naive)]
    # versions = [GEMM('numpy', gemm_numpy, prepare_numpy)
    # versions = [
    #     GEMM('loop_kmn', gemm_kmn, prepare_naive),
    #     GEMM('loop_knm', gemm_knm, prepare_naive),
    #     GEMM('loop_mkn', gemm_mkn, prepare_naive),
    #     GEMM('loop_mnk', gemm_mnk, prepare_naive),
    #     GEMM('loop_nkm', gemm_nkm, prepare_naive),
    #     GEMM('loop_nmk', gemm_nmk, prepare_naive),
    #     ]
    # versions = [GEMM('continuous_memory', gemm_continuous, prepare_continuous)]
    versions = [GEMM('numpy_naive', numpy_naive, prepare_numpy)]

    runs = []

    for version in versions:
        for S in sizes:
            N, M, K = S, S, S
            
            for _ in range(warmup):
                benchmark(version, M, N, K)
            
            
            for i in (pbar := tqdm(range(trials))):
                pbar.set_description(f"Processing {version.name}:{S}")
                
                runs.append({'version': version.name, 'time': benchmark(version, M, N, K), 'size': S} ) 
    
    df = pd.DataFrame(runs)
    df.to_sql("benchmarks", conn, if_exists="append", index=False)


if __name__ == "__main__":
    check_versions()
    # main()
    
