import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def scatterplot(df, x_dim, y_dim):
    x = df[x_dim]
    y = df[y_dim]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(x, y)
    plt.show()


def group_bar_chart(data: dict[str, tuple], y_labels: list[str], y_unit: str, title: str):
    x = np.arange(len(y_labels))  # the label locations
    width = 0.25  # the width of the bars
    multiplier = 0
    fig, ax = plt.subplots(layout='constrained')
    for attribute, measurement in data.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=attribute)
        ax.bar_label(rects, padding=3)
        multiplier += 1
    ax.set_ylabel(y_unit)
    ax.set_title(title)
    ax.set_xticks(x + width, y_labels)
    ax.legend(loc='upper left', ncols=3)


def gen_group_chart_from_csv(csv_path: str, x_column: str, y_columns: list[str]):
    dataframe = pd.read_csv(csv_path)

