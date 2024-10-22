import random
import datetime
import matplotlib.pyplot as plt
import numpy as np
import csv
import codecs
import datetime as dt
import re
import pandas as pd
import os

__author__ = 'Fischbach'
__date__ = '10/22/24'
__assignment = 'MS 3: Data Munging and Visualization'

def mung_data(labels, features):
    # Read the labels file
    labels_df = pd.read_csv(labels,
                            sep=' ',
                            header=None,
                            names=['pass', 'timestamp'],
                            quotechar='"',
                            date_format='%d/%m/%Y %H:%M:%S',
                            parse_dates=[1])

    # Read the features file
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
    print(combined_df.head())


if __name__ == '__main__':
    path_sep = os.path.sep
    labels = 'data_original' + path_sep + 'secom_labels.data'
    features = 'data_original' + path_sep + 'secom.data'
    mung_data(labels, features)
