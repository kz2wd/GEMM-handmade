import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import sqlite3

if __name__ == "__main__":
    
    conn = sqlite3.connect("benchmarks.db")
    df = pd.read_sql_query("SELECT * FROM benchmarks", conn)

    sns.stripplot(data=df, y="time", x="size", hue="version", alpha=.25, legend=None)
    sns.pointplot(data=df, y="time", x="size", hue="version", markers="d", linestyle="none", markersize=4, errorbar=None)
    plt.show()
