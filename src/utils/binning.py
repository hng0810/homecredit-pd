import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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