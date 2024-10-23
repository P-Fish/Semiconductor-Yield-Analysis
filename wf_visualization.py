import os
from dbm import error
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle


def visualize_data():
    pd.options.display.float_format = "{:,.4f}".format

    def find_summary_stats(data):
        def compute_quantitative(df, col):
            mn = df[col].min()
            mx = df[col].max()
            md = df[col].median()
            res = '\nFeature: `' + str(col) + '` - Summary Statistics\n'
            res += 'Min: ' + str(mn) + '\n'
            res += 'Max: ' + str(mx) + '\n'
            res += 'Median: ' + str(md) + '\n'
            return res

        def compute_qualitative(df, col):
            cats = df[col].unique()
            num = len(cats)
            mf = []
            mx = None
            lf = []
            ls = None

            if num == 0:
                raise error

            for cat in cats:
                cur_df = df[df[col] == cat]
                cur_len = len(cur_df)

                if mx is None:
                    mf = [cat]
                    mx = cur_len
                elif cur_len == mx:
                    mf.append(cat)
                elif cur_len > mx:
                    mf = [cat]
                    mx = cur_len

                if ls is None:
                    lf = [cat]
                    ls = cur_len
                elif cur_len == ls:
                    lf.append(cat)
                elif cur_len < ls:
                    lf = [cat]
                    ls = cur_len

            res = '\nFeature: `' + str(col) + '` - Summary Statistics\n'
            res += 'Number of Categories: ' + str(num) + '\n'
            res += 'Most Frequent: ' + ', '.join(str(x) for x in mf) + '\n'
            res += 'Least Frequent: ' + ', '.join(str(x) for x in lf) + '\n'
            return res

        line = '\n' + ('-' * 40) + '\n'

        result = line
        result += 'Qualitative Statistics:\n'
        result += compute_qualitative(data, 'pass')
        result += line

        result += 'Quantitative Statistics:\n'
        for i in range(4):
            result += compute_quantitative(data, 'feature_' + str(i))

        output = (
                data_path
                + 'summary.txt'
        )
        with open(output, 'wb') as output_file:
            try:
                output_file.write(result.encode('utf-8'))
            except Exception as err:
                print(err)
                pass

    def find_pairwise_correlations(data):
        output = (
                data_path
                + 'correlations.txt'
        )

        q_df = data[[col for col in data.columns if col.startswith('feature_')][:4]]
        matrix = q_df.corr()
        half = matrix.where(np.tril(np.ones(matrix.shape)).astype(bool), '')
        print(half)
        with open(output, 'wb') as output_file:
            try:
                output_file.write(str(half).encode('utf-8'))
            except Exception as err:
                print(err)
                pass

    def plot_data(data):
        def plot_scatter(x, y, xlabel, ylabel):
            title = xlabel + ' vs ' + ylabel
            plt.figure()
            plt.title(title)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.grid(True)
            plt.scatter(x, y, s=4)
            plt.savefig(
                visual_path
                + title
                + '.png'
            )

        def plot_histogram():
            title = 'Number of wafers Passed vs Failed'
            plt.figure()
            plt.ylabel('Number of wafers')
            plt.title(title)
            dat = [len(data[data['pass'] == -1]), len(data[data['pass'] == 1])]
            plt.bar(['-1 (Pass)', '1 (Fail)'], dat)
            plt.savefig(
                visual_path
                + title
                + '.png'
            )

        features = [col for col in data.columns if col.startswith('feature_')][:4]
        s_plots = combinations(features, 2)
        for plot in s_plots:
            plot_scatter(data[plot[0]], data[plot[1]], plot[0], plot[1])

        plot_histogram()

    path_sep = os.path.sep
    pickled_data = (
            'data_processed'
            + path_sep
            + 'serialized'
            + path_sep
            + 'secom_output.pickle'
    )

    data_path = (
            'data_processed'
            + path_sep
    )

    visual_path = (
            'data_processed'
            + path_sep
    )

    data_f = None
    with open(pickled_data, 'rb') as f:
        try:
            print('Loading data from pickle file...')
            data_f = pd.DataFrame(pickle.load(f))
        except Exception as err:
            print(err)
            pass

    find_summary_stats(data_f)
    find_pairwise_correlations(data_f)
    plot_data(data_f)


if __name__ == '__main__':
    visualize_data()
