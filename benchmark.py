import sqlite3

import pandas as pd
from tqdm import tqdm

from kernels import benchmark, check, kernels


def main():

    version_names = [
        #   'numpy',
        #   'naive_py', 'numpy_naive',
        #   'loop_kmn', 'loop_knm', 'loop_mkn', 'loop_mnk', 'loop_nkm', 'loop_nmk',
        #   'continuous_memory',
        #   'unrolled2_py','unrolled4_py',
        #   'unrolled8_py', 'unrolled16_py', 'unrolled32_py',
        # "naive_c",
        #   'naive_cu',
        # 'numpy_FP32',
        #   'loop_kmn',
        # 'block_c',
        "kernel_c",
        #'naive_acc_compute',
    ]

    conn = sqlite3.connect("benchmarks.db")

    try:
        df = pd.read_sql_query("SELECT * FROM benchmarks", conn)
    except pd.errors.DatabaseError:
        df = None

    warmup = 0  # check is enough of a warmup
    do_check = True
    trials = 5
    max_data = 100
    # sizes = [512]
    # sizes = [2048]
    sizes = [128, 256, 512, 1024] + list(range(1024 + 512, 4096 + 1, 512))
    runs = []
    try:
        for version_name in version_names:
            try:
                try:
                    version = kernels[version_name]
                except KeyError:
                    print(f"skipping [{version_name}] (unrecognized)", end=" ")
                    continue
                if do_check:
                    check(version)
                for K in sizes:
                    if df is not None:
                        size_filtered_df = df[df["size"] == K]
                        if (
                            size_filtered_df[
                                size_filtered_df["version"] == version_name
                            ].count()
                            >= max_data
                        ).any():
                            print(
                                f"skipping [{version.name}][{K}] (max data_reached)",
                                end=" ",
                            )
                            continue

                    for _ in range(warmup):
                        benchmark(version, K)

                    for i in (pbar := tqdm(range(trials))):
                        pbar.set_description(f"Processing {version.name}:{K}")

                        runs.append(
                            {
                                "version": version_name,
                                "name": version.name,
                                "time": benchmark(version, K),
                                "size": K,
                                "precision": version.precision,
                                "device": version.device,
                                "language": version.language,
                            }
                        )

            except RuntimeError as e:
                print(e)
        print("Done")
    except KeyboardInterrupt:
        print("User cancelled")
    print("Saving runs")
    df_save = pd.DataFrame(runs)
    df_save.to_sql("benchmarks", conn, if_exists="append", index=False)


if __name__ == "__main__":
    main()
