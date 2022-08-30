import pandas as pd
import numpy as np
import math
FILE_NAME = 'shuttle'
target_col = "9"
WINDOW_SIZE = 10000
# in percentage, e.g. 0.96%
ANOMALY_RATE = 0.96

dataset_path = f'./datasets/{FILE_NAME}.csv'
output_path = f'./datasets/{FILE_NAME}_small.csv'
MIN_OUTLIERS = math.ceil(WINDOW_SIZE * (ANOMALY_RATE / 100))
df = pd.read_csv(dataset_path)
df['rolling_sum'] = df[target_col].rolling(WINDOW_SIZE).sum()
try:
    end_index = df[df['rolling_sum'] > MIN_OUTLIERS].index[0] + 1
except:
    print('Emnpty index, please try smaller min number of outliers...')
df = df.drop(['rolling_sum'], axis = 1)
df = df.iloc[(end_index - WINDOW_SIZE):end_index, :].reset_index(drop=True)
print(df[target_col].sum())
assert df[target_col].sum() >= MIN_OUTLIERS
df.to_csv(output_path, index=False)
print('outlier sum: ', df[target_col].sum(), ' index: ', end_index)