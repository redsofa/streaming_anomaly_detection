'''
Note that this code is simple and repetitive on purpose.
It's meant to describe datasets for a paper at a high level and to be thrown away.

Console output of this script :

---------
Data Directory is : /home/richardr/dev/git/Online-Outlier-Detection/datasets

##################################
Shuttle dataset statistics ... 

Raw data sample ...
---
    0  1   2  3   4   5   6   7   8  9
0  38  1  96  0  38  13  58  57   0  0
1  37  0  77  0  24  25  40  54  14  0
2  45  0  83  0  44 -17  38  39   2  0
3  56 -1  84  0  54 -30  28  30   2  0
4  55  5  77  0  54   0  23  23   0  0

Summary of dataset ...
---
Number of Rows : 49097
Number of Columns : 10

Label column details ... 
Normal = 0 and outlier = 1 ... 
---
0    45586
1     3511
Name: 9, dtype: int64

Outlier percentage ...
---
7.151149764751411
##################################

##################################
Http dataset statistics ... 

Raw data sample ...
---
   0    1      2  3
0  0  215  45076  0
1  0  162   4528  0
2  0  236   1228  0
3  0  233   2032  0
4  0  239    486  0

Summary of dataset ...
---
Number of Rows : 567497
Number of Columns : 4

Label column details ... 
Normal = 0 and outlier = 1 ... 
---
0    565286
1      2211
Name: 3, dtype: int64

Outlier percentage ...
---
0.38960558381806426
##################################

##################################
smtp dataset statistics ... 

Raw data sample ...
---
   0     1    2  3
0  1  1207  329  0
1  1  1679  333  0
2  0   979  330  0
3  1  1535  325  0
4  1  1706  326  0

Summary of dataset ...
---
Number of Rows : 95156
Number of Columns : 4

Label column details ... 
Normal = 0 and outlier = 1 ... 
---
0    95126
1       30
Name: 3, dtype: int64

Outlier percentage ...
---
0.03152717642607928
##################################

##################################
ForestCover dataset statistics ... 

Raw data sample ...
---
      0    1   2    3    4     5    6    7    8     9  10
0  2804  139   9  268   65  3180  234  238  135  6121   0
1  2785  155  18  242  118  3090  238  238  122  6211   0
2  2579  132   6  300  -15    67  230  237  140  6031   0
3  2886  151  11  371   26  5253  234  240  136  4051   0
4  2742  134  22  150   69  3215  248  224   92  6091   0

Summary of dataset ...
---
Number of Rows : 286048
Number of Columns : 11

Label column details ... 
Normal = 0 and outlier = 1 ... 
---
0    283301
1      2747
Name: 10, dtype: int64

Outlier percentage ...
---
0.9603283365029646
##################################

##################################
Bank dataset statistics ... 

Raw data sample ...
---
        age  duration  campaign  previous  ...  cons.conf.idx  euribor3m  nr.employed  class
0  0.209877  0.027654  0.127273  0.000000  ...       0.376569   0.980730     1.000000      0
1  0.296296  0.017080  0.072727  0.000000  ...       0.615063   0.981183     1.000000      0
2  0.246914  0.028060  0.090909  0.000000  ...       0.602510   0.957379     0.859735      0
3  0.160494  0.043310  0.018182  0.142857  ...       0.192469   0.150759     0.512287      0
4  0.530864  0.110817  0.000000  0.000000  ...       0.154812   0.174790     0.512287      1

[5 rows x 10 columns]

Summary of dataset ...
---
Number of Rows : 41188
Number of Columns : 10

Label column details ... 
Normal = 0 and outlier = 1 ... 
---
0    36548
1     4640
Name: class, dtype: int64

Outlier percentage ...
---
11.265417111780131
##################################

##################################
credit card fraud dataset statistics ... 

Raw data sample ...
---
         V1        V2        V3        V4        V5  ...       V26       V27       V28    Amount  class
0  0.935192  0.766490  0.881365  0.313023  0.763439  ...  0.394557  0.418976  0.312697  0.005824      0
1  0.978542  0.770067  0.840298  0.271796  0.766120  ...  0.446013  0.416345  0.313423  0.000105      0
2  0.935217  0.753118  0.868141  0.268766  0.762329  ...  0.402727  0.415489  0.311911  0.014739      0
3  0.941878  0.765304  0.868484  0.213661  0.765647  ...  0.389197  0.417669  0.314371  0.004807      0
4  0.938617  0.776520  0.864251  0.269796  0.762975  ...  0.507497  0.420561  0.317490  0.002724      0

[5 rows x 30 columns]

Summary of dataset ...
---
Number of Rows : 284807
Number of Columns : 30

Label column details ... 
Normal = 0 and outlier = 1 ... 
---
0    284315
1       492
Name: class, dtype: int64

Outlier percentage ...
---
0.1727485630620034
##################################

##################################
credit card fraud dataset statistics ... 

Raw data sample ...
---
         f1        f2  label
0 -0.148190  1.058867      0
1 -0.040583  1.223463      0
2  0.387559  1.101300      0
3  0.966357  1.100466      0
4  0.311854  0.996458      0

Summary of dataset ...
---
Number of Rows : 25006
Number of Columns : 3

Label column details ... 
Normal = 0 and outlier = 1 ... 
---
0    22614
1     2392
Name: label, dtype: int64

Outlier percentage ...
---
9.565704230984563
##################################
---------
'''

from globals import SETTINGS
import pandas as pd

DATA_DIR = SETTINGS['DATA_DIR']

def read_shuttle():
    df = pd.read_csv(
        filepath_or_buffer = '{}/Shuttle.csv'.format(DATA_DIR),
        sep=',',
        header="infer")

    return df


def process_shuttle(df):
    print('##################################')
    print('Shuttle dataset statistics ... ')
    print()
    print('Raw data sample ...')
    print('---')
    print(df.head())

    print()
    print('Summary of dataset ...')
    print('---')
    number_of_rows = df.shape[0]
    print('Number of Rows : {}'.format(number_of_rows))
    print('Number of Columns : {}'.format(df.shape[1]))

    print()
    print('Label column details ... ')
    print('Normal = 0 and outlier = 1 ... ')
    print('---')
    print(df['9'].value_counts())
    normal = df['9'].value_counts()[0]
    outliers =df['9'].value_counts()[1]
    print()
    print('Outlier percentage ...')
    print('---')
    print(outliers/number_of_rows * 100)
    print('##################################')


def read_http():
    df = pd.read_csv(
        filepath_or_buffer = '{}/HTTP.csv'.format(DATA_DIR),
        sep=',',
        header="infer")

    return df


def process_http(df):
    print('##################################')
    print('Http dataset statistics ... ')
    print()
    print('Raw data sample ...')
    print('---')
    print(df.head())

    print()
    print('Summary of dataset ...')
    print('---')
    number_of_rows = df.shape[0]
    print('Number of Rows : {}'.format(number_of_rows))
    print('Number of Columns : {}'.format(df.shape[1]))

    print()
    print('Label column details ... ')
    print('Normal = 0 and outlier = 1 ... ')
    print('---')
    print(df['3'].value_counts())
    normal = df['3'].value_counts()[0]
    outliers =df['3'].value_counts()[1]
    print()
    print('Outlier percentage ...')
    print('---')
    print(outliers/number_of_rows * 100)
    print('##################################')


def read_smtp():
    df = pd.read_csv(
        filepath_or_buffer = '{}/SMTP.csv'.format(DATA_DIR),
        sep=',',
        header="infer")

    return df


def process_smtp(df):
    print('##################################')
    print('smtp dataset statistics ... ')
    print()
    print('Raw data sample ...')
    print('---')
    print(df.head())

    print()
    print('Summary of dataset ...')
    print('---')
    number_of_rows = df.shape[0]
    print('Number of Rows : {}'.format(number_of_rows))
    print('Number of Columns : {}'.format(df.shape[1]))

    print()
    print('Label column details ... ')
    print('Normal = 0 and outlier = 1 ... ')
    print('---')
    print(df['3'].value_counts())
    normal = df['3'].value_counts()[0]
    outliers =df['3'].value_counts()[1]
    print()
    print('Outlier percentage ...')
    print('---')
    print(outliers/number_of_rows * 100)
    print('##################################')


def read_forest_cover():
    df = pd.read_csv(
        filepath_or_buffer = '{}/ForestCover.csv'.format(DATA_DIR),
        sep=',',
        header="infer")

    return df


def process_forest_cover(df):
    print('##################################')
    print('ForestCover dataset statistics ... ')
    print()
    print('Raw data sample ...')
    print('---')
    print(df.head())

    print()
    print('Summary of dataset ...')
    print('---')
    number_of_rows = df.shape[0]
    print('Number of Rows : {}'.format(number_of_rows))
    print('Number of Columns : {}'.format(df.shape[1]))

    print()
    print('Label column details ... ')
    print('Normal = 0 and outlier = 1 ... ')
    print('---')
    print(df['10'].value_counts())
    normal = df['10'].value_counts()[0]
    outliers =df['10'].value_counts()[1]
    print()
    print('Outlier percentage ...')
    print('---')
    print(outliers/number_of_rows * 100)
    print('##################################')


def read_bank():
    df = pd.read_csv(
        filepath_or_buffer = '{}/bank.csv'.format(DATA_DIR),
        sep=',',
        header="infer")

    return df


def process_bank(df):
    print('##################################')
    print('Bank dataset statistics ... ')
    print()
    print('Raw data sample ...')
    print('---')
    print(df.head())

    print()
    print('Summary of dataset ...')
    print('---')
    number_of_rows = df.shape[0]
    print('Number of Rows : {}'.format(number_of_rows))
    print('Number of Columns : {}'.format(df.shape[1]))

    print()
    print('Label column details ... ')
    print('Normal = 0 and outlier = 1 ... ')
    print('---')
    print(df['class'].value_counts())
    normal = df['class'].value_counts()[0]
    outliers =df['class'].value_counts()[1]
    print()
    print('Outlier percentage ...')
    print('---')
    print(outliers/number_of_rows * 100)
    print('##################################')


def read_ccard_fraud():
    df = pd.read_csv(
    filepath_or_buffer = '{}/creditcardfraud_normalised.csv'.format(DATA_DIR),
    sep=',',
    header="infer")

    return df


def process_ccard_fraud(df):
    print('##################################')
    print('credit card fraud dataset statistics ... ')
    print()
    print('Raw data sample ...')
    print('---')
    print(df.head())

    print()
    print('Summary of dataset ...')
    print('---')
    number_of_rows = df.shape[0]
    print('Number of Rows : {}'.format(number_of_rows))
    print('Number of Columns : {}'.format(df.shape[1]))

    print()
    print('Label column details ... ')
    print('Normal = 0 and outlier = 1 ... ')
    print('---')
    print(df['class'].value_counts())
    normal = df['class'].value_counts()[0]
    outliers =df['class'].value_counts()[1]
    print()
    print('Outlier percentage ...')
    print('---')
    print(outliers/number_of_rows * 100)
    print('##################################')


def read_generated():
    df = pd.read_csv(
    filepath_or_buffer = '{}/generated_25000_samples_anomalies.csv'.format(DATA_DIR),
    sep=',',
    header="infer")

    return df

def process_generated(df):
    print('##################################')
    print('credit card fraud dataset statistics ... ')
    print()
    print('Raw data sample ...')
    print('---')
    print(df.head())

    print()
    print('Summary of dataset ...')
    print('---')
    number_of_rows = df.shape[0]
    print('Number of Rows : {}'.format(number_of_rows))
    print('Number of Columns : {}'.format(df.shape[1]))

    print()
    print('Label column details ... ')
    print('Normal = 0 and outlier = 1 ... ')
    print('---')
    print(df['label'].value_counts())
    normal = df['label'].value_counts()[0]
    outliers =df['label'].value_counts()[1]
    print()
    print('Outlier percentage ...')
    print('---')
    print(outliers/number_of_rows * 100)
    print('##################################')


def main():
    print('Data Directory is : {}'.format(DATA_DIR))
    print()
    df = read_shuttle()
    process_shuttle(df)

    print()
    df = read_http()
    process_http(df)

    print()
    df = read_smtp()
    process_smtp(df)

    print()
    df = read_forest_cover()
    process_forest_cover(df)

    print()
    df = read_bank()
    process_bank(df)

    print()
    df = read_ccard_fraud()
    process_ccard_fraud(df)

    print()
    df = read_generated()
    process_generated(df)


if __name__ == '__main__':
    print('---------')
    main()
    print('---------')