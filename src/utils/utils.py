
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math

def _calculate_missing_percentage(df=pd.DataFrame()):
    missing_value_percentage = []
    for col in df.columns:
        missing_value_percentage.append((col, df[col].isnull().sum() / len(df)))
    missing_value_percentage = sorted(missing_value_percentage, key=lambda x: x[1], reverse=True)
    missing_value_percentage = pd.DataFrame(missing_value_percentage, columns=['Feature', 'Missing Value Percentage'])
    return missing_value_percentage
# Vẽ box_plot
def _plot_boxplot(df, column):
    plt.figure(figsize=(20, 6))
    sns.boxplot(x=df[column])
    plt.title(f'Box Plot of {column}')
    plt.xlabel(column)
    plt.show()
# Vẽ scatter plot
def _plot_scatter(df, x_column, y_column, z_column=None):
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

def _plot_bar(df, column):
    plt.figure(figsize=(20, 6))
    sns.countplot(x=df[column])
    plt.title(f'Bar Plot of {column}')
    plt.xlabel(column)
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.show()

# Biểu đồ histogram
def _plot_histogram(df, column, bins=10, use_kde=True):
    plt.figure(figsize=(20, 6))
    x = df[column].dropna()
    xlabel = f'{column} bin ticks'
    ylabel = f'Count'
    title = f'Histogram of {column} with {bins} bins'
    ax = sns.histplot(x, bins=bins, kde=use_kde)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    return ax

def _plot_bar_hue(df, column, hue_column):
    plt.figure(figsize=(20, 6))
    sns.countplot(data=df, x=column, hue=hue_column)
    plt.title(f'Bar Plot of {column}')
    plt.xlabel(f'{column} bin ticks')
    plt.ylabel('Count')
    plt.legend(title=f'{hue_column}', loc='upper right')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Multi plotting
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

def _plot_auto_grid(df, columns=None, bins=10, use_kde=True):
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

MAX_VAL = np.inf
MIN_VAL = -MAX_VAL

def _bin_table(df, target, feature, n_bins=6, qcut=None):
    """
    df: DataFrame of features
    target: Series of target (good_bad)
    feature: feature name to bin
    n_bins: number of bins for qcut, default = 6
    qcut: custom bin edges, default = None
    """
    bin_data = df[[feature]].copy()
    bin_data['target'] = target.values
    coltype = bin_data[feature].dtype

    # Numeric binning
    if coltype not in ['object']:
        if qcut is None:
            try:
                # Không nhập qcut thì cắt bằng n_bins
                bins, thres = pd.qcut(bin_data[feature], q=n_bins, retbins=True, duplicates='drop')
                thres[0] = MIN_VAL
                thres[-1] = MAX_VAL
                bin_data['bins'] = pd.cut(bin_data[feature], bins=thres, include_lowest=True)
            except Exception as e:
                # Cắt n_bins có thể failed do giá trị tập trung vào 1 điểm
                print('qcut failed - n_bins must be lower to bin interval is valid!', e)
                return None, None
        else:
            # Cắt tay bằng qcut bin
            bin_data['bins'] = pd.cut(bin_data[feature], bins=qcut, include_lowest=True)
            thres = qcut

    # Object binning, bin bằng giá trị unique trong bộ dữ liệu
    elif coltype == 'object':
        bin_data['bins'] = bin_data[feature]
        thres = None
    else:
        raise ValueError("Unsupported feature type for binning.")
    # Sau khi cắt xong sẽ ném hết binning index vào cột bin_data['bins']

    # Xây dựng bảng kết quả binning
    result_table = bin_data.groupby('bins')['target'].value_counts().unstack(fill_value=0) # Nếu 1 bin không có target class nào thì count = 0 luôn
    if 0 not in result_table.columns:
        result_table[0] = 0
    if 1 not in result_table.columns:
        result_table[1] = 0
    result_table = result_table[[0, 1]]
    result_table.columns = ['#GOOD', '#BAD']
    result_table['No_Obs'] = result_table['#BAD'] + result_table['#GOOD']
    # Ném thres ngược vào bảng result cho dễ nhìn
    if thres is not None:
        df_Thres = pd.DataFrame({'Thres': thres[1:]}, index=result_table.index)
        df_summary = df_Thres.join(result_table)
    else:
        df_summary = result_table
    # Đưa các chỉ số còn lại vào
    df_summary['COLUMN'] = feature
    df_summary['GOOD/BAD'] = df_summary['#GOOD']/df_summary['#BAD']
    df_summary['%BAD'] = df_summary['#BAD']/df_summary['#BAD'].sum()
    df_summary['%GOOD'] = df_summary['#GOOD']/df_summary['#GOOD'].sum()
    df_summary['WOE'] = np.log(df_summary['%GOOD']/df_summary['%BAD'])
    df_summary['IV'] = (df_summary['%GOOD']-df_summary['%BAD'])*df_summary['WOE']
    IV = df_summary['IV'].sum()
    print('Information Value of {} column: {}'.format(feature, IV))
    return df_summary, IV, thres

# Vẽ biểu đồ WOE
def _woe_plot(df_summary, IV):
    colname = list(df_summary['COLUMN'].unique())[0]
    df_summary['WOE'].plot(linestyle='-', marker='o')
    plt.title('WOE of {} field. IV = {}'.format(colname, IV))
    plt.axhline(y=0, color = 'red')
    plt.xticks(rotation=45)
    plt.ylabel('WOE')
    plt.xlabel('Bin group')
    plt.show()