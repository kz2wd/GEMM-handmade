import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import sqlite3

if __name__ == "__main__":
    
    conn = sqlite3.connect("benchmarks.db")
    versions_to_plot = []
    versions_to_plot = ["numpy", "unrolled4_py", 'unrolled8_py', 'unrolled16_py', 'unrolled32_py']

    if not versions_to_plot:
        df = pd.read_sql_query("SELECT * FROM benchmarks", conn)
    else:
        placeholder = ", ".join('?' for _ in versions_to_plot) 
        query = f"SELECT * FROM benchmarks WHERE version IN ({placeholder})"
        df = pd.read_sql_query(query, conn, params = versions_to_plot)


    sns.stripplot(data=df, y="time", x="size", hue="version", alpha=.25, legend=None)
    sns.pointplot(data=df, y="time", x="size", hue="version", markers="d", linestyle="none", markersize=4, errorbar=None)
    plt.show()


