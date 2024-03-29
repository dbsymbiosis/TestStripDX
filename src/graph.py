import logging
import os.path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def scatterplot(df, x_dim, y_dim):
    x = df[x_dim]
    y = df[y_dim]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(x, y)
    plt.show()


def group_bar_chart(data: dict[str, tuple], x_labels: list[str], y_unit: str, title: str, save_file_path: str, img_h,
                    img_w):
    x = np.arange(len(x_labels))  # the label locations
    width = 0.25  # the width of the bars
    multiplier = 0
    fig, ax = plt.subplots(layout='constrained', figsize=[img_w, img_h])
    for attribute, measurement in data.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=attribute)
        ax.bar_label(rects, padding=3)
        multiplier += 1
    ax.set_ylabel(y_unit)
    ax.set_title(title)
    ax.set_xticks(x + width, x_labels, rotation='vertical')
    ax.legend(loc='upper left', ncols=3)
    plt.savefig(save_file_path)


def gen_save_group_chart_from_csv(csv_path: str, x_column: str, y_columns: list[str], output_dir: str, graph_h,
                                  graph_w):
    logging.info(f'Plotting data from file {csv_path}')
    save_file_path = output_dir + '/Result.png'
    if not os.path.exists(output_dir):
        logging.error('The given output directory does not exist. Please provide a valid path to save the graph.')
        sys.exit(0)
    try:
        dataframe = pd.read_csv(csv_path)
    except:
        logging.error('The csv file does not exist. Please provide a valid file path')
        sys.exit(0)
    if x_column not in dataframe.columns:
        logging.error(f'Invalid column name provided:{x_column}. Please provide a valid column name')
        sys.exit(0)
    for y_col in y_columns:
        if y_col not in dataframe.columns:
            logging.error(f'Invalid column name provided:{y_col}. Please provided a valid column name')
    dataframe = dataframe.round(2)
    data = {}
    for key in y_columns:
        data[key] = dataframe[key].values
    group_bar_chart(data, dataframe[x_column], 'Color intensity', 'Color Intensity', save_file_path,graph_h,
                    graph_w)
    logging.info(f'Data from {csv_path} is plotted and the graph is saved to file {save_file_path}')
