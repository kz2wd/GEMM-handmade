import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3

if __name__ == "__main__":

    conn = sqlite3.connect("benchmarks.db")
    df = pd.read_sql_query("SELECT * FROM benchmarks", conn)

    sns.barplot(data=df, y="time", x="size", hue="version")
    plt.show()
