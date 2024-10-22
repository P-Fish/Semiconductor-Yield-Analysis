import numpy as np
import pandas as pd
import pickle
import os

__author__ = 'Fischbach'
__date__ = '10/22/24'
__assignment = 'MS 3: Data Munging and Visualization'


def mung_data():

    def simulate_nan_values(df):
        result_df = df.copy()
        feature_cols = [col for col in df.columns if col.startswith('feature_')]

        for value in df['pass'].unique():
            mask = df['pass'] == value
            class_df = df[mask]

            for col in feature_cols:
                if df[col].isna().any():
                    median_val = class_df[col].median()
                    std_val = class_df[col].std()

                    nan_count = df.loc[mask, col].isna().sum()
                    if nan_count > 0 and not pd.isna(median_val) and not pd.isna(std_val):
                        rng = np.random.default_rng()
                        random_values = rng.normal(
                            loc=median_val,
                            scale=std_val,
                            size=nan_count
                        )
                        result_df.loc[mask & df[col].isna(), col] = random_values

        return result_df


    path_sep = os.path.sep
    labels = 'data_original' + path_sep + 'secom_labels.data'
    features = 'data_original' + path_sep + 'secom.data'
    output = 'data_processed' + path_sep + 'secom_output.pickle'

    labels_df = pd.read_csv(labels,
                            sep=' ',
                            header=None,
                            names=['pass', 'timestamp'],
                            quotechar='"',
                            date_format='%d/%m/%Y %H:%M:%S',
                            parse_dates=[1])

    features_df = pd.read_csv(features,
                              sep=' ',
                              header=None,
                              names=[f'feature_{i}' for i in range(590)],
                              na_values='NaN',
                              dtype=np.float64,
                              index_col=False)

    labels_df = labels_df.reset_index(drop=True)
    features_df = features_df.reset_index(drop=True)

    combined_df = pd.concat([labels_df, features_df], axis=1, ignore_index=False)

    final_df =  simulate_nan_values(combined_df)

    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, 'wb') as output_file:
        try:
            pickle.dump(final_df, output_file)
        except Exception as err:
            print(err)
            pass


if __name__ == '__main__':
    mung_data()