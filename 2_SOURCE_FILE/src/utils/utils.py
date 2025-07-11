import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Dev simple functions
# Tính tỷ lệ giá trị khuyết thiếu
def calculate_missing_percentage(df=pd.DataFrame()):
    missing_value_percentage = []
    for col in df.columns:
        missing_value_percentage.append((col, df[col].isnull().sum() / len(df)))
    missing_value_percentage = sorted(missing_value_percentage, key=lambda x: x[1], reverse=True)
    missing_value_percentage = pd.DataFrame(missing_value_percentage, columns=['Feature', 'Missing Value Percentage'])
    return missing_value_percentage
# Vẽ box_plot
def plot_boxplot(df, column):
    plt.figure(figsize=(20, 6))
    sns.boxplot(x=df[column])
    plt.title(f'Box Plot of {column}')
    plt.xlabel(column)
    plt.show()
# Vẽ scatter plot
def plot_scatter(df, x_column, y_column, z_column=None):
    if z_column:
        plt.figure(figsize=(20, 6))
        sns.scatterplot(x=df[x_column], y=df[y_column], hue=df[z_column])
        plt.title(f'Scatter Plot of {x_column} vs {y_column} colored by {z_column}')
        plt.xlabel(x_column)
        plt.ylabel(y_column)
        plt.legend(title=z_column)
        plt.show()
    else:
        if x_column == y_column:
            raise ValueError("x_column and y_column cannot be the same for a scatter plot.")
# Vẽ bar plot
def plot_bar(df, column):
    plt.figure(figsize=(20, 6))
    sns.countplot(x=df[column])
    plt.title(f'Bar Plot of {column}')
    plt.xlabel(column)
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.show()
# Draw multiple plots in one figure
# Biểu đồ histogram
# def _plot_hist_subplot(x, fieldname, bins = 10, use_kde = True):
#   x = x.dropna()
#   xlabel = '{} bins tickers'.format(fieldname)
#   ylabel = 'Count obs in {} each bin'.format(fieldname)
#   title = 'histogram plot of {} with {} bins'.format(fieldname, bins)
#   ax = sns.distplot(x, bins = bins, kde = use_kde)
#   ax.set_xlabel(xlabel)
#   ax.set_ylabel(ylabel)
#   ax.set_title(title)
#   return ax
# # Biểu đồ barchart
# def _plot_barchart_subplot(x, fieldname):
#   xlabel = 'Group of {}'.format(fieldname)
#   ylabel = 'Count obs in {} each bin'.format(fieldname)
#   title = 'Barchart plot of {}'.format(fieldname)
#   x = x.fillna('Missing')
#   df_summary = x.value_counts(dropna = False)
#   y_values = df_summary.values
#   x_index = df_summary.index
#   ax = sns.barplot(x = x_index, y = y_values, order = x_index)
#   # Tạo vòng for lấy tọa độ đỉnh trên cùng của biểu đồ và thêm label thông qua annotate.
#   labels = list(set(x))
#   for label, p in zip(y_values, ax.patches):
#     ax.annotate(label, (p.get_x()+0.25, p.get_height()+0.15))
#   plt.xlabel(xlabel)
#   plt.ylabel(ylabel)
#   plt.title(title) # 
#   return ax

# Multi plotting
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math

def _plot_hist_subplot(ax, x, fieldname, bins=10, use_kde=True):
    x = x.fillna(999999)
    xlabel = f'{fieldname} bins tickers'
    ylabel = f'Count obs in {fieldname} each bin'
    title = f'Histogram of {fieldname} ({bins} bins)'
    sns.histplot(x, bins=bins, kde=use_kde, ax=ax)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

def _plot_hist_subplot_binary(ax, x, fieldname, bins=2, use_kde=True):
    x = x.fillna(999999)
    xlabel = f'{fieldname} bins tickers'
    ylabel = f'Count obs in {fieldname} each bin'
    title = f'Histogram of {fieldname} ({bins} bins)'
    sns.histplot(x, bins=bins, kde=use_kde, ax=ax)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

def _plot_barchart_subplot(ax, x, fieldname):
    x = x.fillna('Missing')
    df_summary = x.value_counts(dropna=False)
    y_values = df_summary.values
    x_index = df_summary.index

    sns.barplot(x=x_index, y=y_values, order=x_index, ax=ax)
    for label, p in zip(y_values, ax.patches):
        ax.annotate(label, (p.get_x() + p.get_width() / 4, p.get_height() + 0.15))

    ax.set_xlabel(f'Group of {fieldname}')
    ax.set_ylabel(f'Count obs in {fieldname}')
    ax.set_title(f'Barchart of {fieldname}')
    ax.tick_params(axis='x', rotation=45)

def plot_auto_grid(df, columns=None, bins=10, use_kde=True):
    if columns is None:
        columns = df.columns.tolist()

    valid_columns = []
    plot_types = []

    for col in columns:
        series = df[col]
        dtype = series.dtype

        if pd.api.types.is_numeric_dtype(dtype):
            valid_columns.append(col)
            plot_types.append('hist')

        elif pd.api.types.is_object_dtype(dtype) or pd.api.types.is_categorical_dtype(dtype):
            n_unique = series.nunique(dropna=False)
            if n_unique <= 10:
                valid_columns.append(col)
                plot_types.append('bar')
            else:
                print(f"Skipping column '{col}': too many groups ({n_unique})")

        else:
            print(f"Skipping column '{col}': unsupported data type ({dtype})")

    n_plots = len(valid_columns)
    if n_plots == 0:
        print("No valid columns to plot.")
        return

    n_cols = 4
    n_rows = math.ceil(n_plots / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = axes.flatten()

    for idx, (col, ptype) in enumerate(zip(valid_columns, plot_types)):
        ax = axes[idx]
        series = df[col]

        if ptype == 'hist':
            _plot_hist_subplot(ax, series, col, bins=bins, use_kde=use_kde)
        elif ptype == 'bar':
            _plot_barchart_subplot(ax, series, col)

    # Xóa subplot thừa nếu có
    for j in range(idx + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()