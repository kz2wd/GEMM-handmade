import sqlite3

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def fetch_df(versions_to_plot):
    conn = sqlite3.connect("benchmarks.db")

    if not versions_to_plot:
        df = pd.read_sql_query("SELECT * FROM benchmarks", conn)
    else:
        placeholder = ", ".join("?" for _ in versions_to_plot)
        query = f"SELECT * FROM benchmarks WHERE version IN ({placeholder})"
        df = pd.read_sql_query(query, conn, params=versions_to_plot)

    return df


# I have a i5-8600K / 3070ti
# FLOPS = cores * frequency * FLOP per cycle
# i5-8600K
# FLOP per cycle : 2 * 256-bit FMA -> 16 FP64 / 32 FP32
# per core : 4.3 GHz * 16 = 68.8 / 137.6 GFLOPS
# total : 6 cores -> 412.8 / 825.6 GFLOPS
# 4.3 GHz is turbo mode but 3.6 GHz would be fairer for multi core sustained operations.
# sustained: /core = 57.6 / 115.2
#            total = 345.6 / 691.2

# 3070 ti https://www.techpowerup.com/gpu-specs/geforce-rtx-3070-ti.c3675
# FP64 339.8 GFLOPS
# FP32 21.75 TFLOPS

# GEMM FLOP -> 2K**3 - K**2 -> 2K**3
# FLOPS of run: GEMM FLOP / run time


def plot_flops_global():

    versions_to_plot = []
    # versions_to_plot = ["numpy", 'naive_c', 'unrolled32_py']
    df = fetch_df(versions_to_plot)
    df["GFLOPS"] = df.apply(
        lambda row: (2 * row["size"] ** 3) / row["time"] * 1e-9, axis=1
    )
    sns.stripplot(data=df, y="GFLOPS", x="size", hue="name", alpha=0.25, legend=None)
    sns.pointplot(
        data=df,
        y="GFLOPS",
        x="size",
        hue="name",
        markers="d",
        linestyle="none",
        markersize=4,
        errorbar=None,
    )
    plt.title("Global GFLOPS")
    plt.show()


def plot_flops_cpu():

    versions_to_plot = []
    versions_to_plot = ["numpy", "naive_c", "unrolled32_py", "numpy_FP32"]
    df = fetch_df(versions_to_plot)
    df["GFLOPS"] = df.apply(
        lambda row: (2 * row["size"] ** 3) / row["time"] * 1e-9, axis=1
    )
    sns.stripplot(data=df, y="GFLOPS", x="size", hue="name", alpha=0.25, legend=None)
    sns.pointplot(
        data=df,
        y="GFLOPS",
        x="size",
        hue="name",
        markers="d",
        linestyle="none",
        markersize=4,
        errorbar=None,
    )
    plt.title("Global GFLOPS")

    plt.axhline(y=412.8, label="FP64 GFLOPS LIMIT")
    plt.axhline(y=345.6, label="FP64 GFLOPS SUSTAIN")
    plt.axhline(y=412.8 * 2, label="FP32 GFLOPS LIMIT")
    plt.legend()
    plt.show()


def plot_smaller():
    versions_to_plot = ["naive_c", "block_c", "kernel_c"]
    df = fetch_df(versions_to_plot)
    df["GFLOPS"] = df.apply(
        lambda row: (2 * row["size"] ** 3) / row["time"] * 1e-9, axis=1
    )
    sns.stripplot(data=df, y="GFLOPS", x="size", hue="name", alpha=0.25, legend=None)
    sns.pointplot(
        data=df,
        y="GFLOPS",
        x="size",
        hue="name",
        markers="d",
        linestyle="none",
        markersize=4,
        errorbar=None,
        dodge=True,
    )
    plt.axhline(y=57.6, label="FP64 GFLOPS LIMIT")
    plt.title("Global GFLOPS")
    plt.show()


def plot_times():

    versions_to_plot = []
    # versions_to_plot = ["numpy", 'naive_c', 'unrolled32_py']
    df = fetch_df(versions_to_plot)
    sns.stripplot(data=df, y="time", x="size", hue="name", alpha=0.25, legend=None)
    sns.pointplot(
        data=df,
        y="time",
        x="size",
        hue="name",
        markers="d",
        linestyle="none",
        markersize=4,
        errorbar=None,
    )
    plt.show()


if __name__ == "__main__":
    # plot_times()
    plot_smaller()
