import numpy as np
import pandas as pd
from time import perf_counter
from tqdm import tqdm
from gemm.gemm_naive import gemm_naive, prepare_naive
from gemm.gemm_numpy import gemm_numpy, prepare_numpy
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



def main():
    conn = sqlite3.connect("benchmarks.db")


    sizes = [128, 256, 512, 1024]

    warmup = 1
    trials = 10
    # GEMM('naive_py', gemm_naive, prepare_naive)
    versions = [GEMM('numpy', gemm_numpy, prepare_numpy)]

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
    main()
    