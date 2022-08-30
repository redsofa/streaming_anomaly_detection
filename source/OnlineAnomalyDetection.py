"""
Importing all required packages
"""

from AELSTMOnline import AELSTMOnline
from skmultiflow.data import FileStream
import pandas as pd
from datetime import datetime
import argparse
from skmultiflow.anomaly_detection import HalfSpaceTrees
from scikit_multiflow_iForestASD import *
from sklearn import preprocessing
from sklearn.neighbors import LocalOutlierFactor
import time
from baseAnomalyDetector import *
import os
from tensorflow.keras import backend as K
from iforestasd_scikitmultiflow_PADWIN import *

if not os.path.exists(f'./results'):
    os.mkdir(f'./results')
LOF_N_NEIGHBORS = 2
LOF_CONTAMINATION = 0.1
PCA_COMP = 2
N_NEURONS = 256
N_EPOCHS = 2
# start from 0
TIME_SERIES_DRAW_LEFT_IDX = 1
TIME_SERIES_DRAW_RIGHT_IDX = 2
SHOW_PLOT = False
SHOW_PCA = SHOW_PLOT
SHOW_TIME_SERIES = False
MAD_CL = 0.95
results_to_saved = []
def start_training_banner(name):
    print("**********************************")
    print(f"Anamoly detection for local data stream using {name}")
    print("**********************************")

def print_result(_model, name, y_voted):
    current_result = dict()
    start_training_banner(name)
    if _model.isPretrain:
        print(f"Pre-training of unsupervised {_model.name}  model with {_model.initial_batch_training_size} instances")

    print("Total length of data found: ", _model.len_of_data)
    print("The number of anomaly found: ", _model.get_num_of_anomaly())
    # if there is no true value, use the voted results
    if _model.y_true == []:
        _model.y_true = y_voted

    if len(set(_model.y_true)) == 1:
        print("**********************************")
        print("There is only one class (either all 0 or all 1), so we are not able to print the result")
        print("**********************************")
        return

    res_dict = _model.results()
    if _model.end_time != None:
        print(f"Total Time for {_model.name} " + str(_model.end_time - _model.start_time))
        res_dict['time'] = str(_model.end_time - _model.start_time)
    results_to_saved.append(res_dict)
def get_file_stream(data_source, target_column):
    if target_column == None:
        stream = FileStream(data_source, allow_nan=True, n_targets=0)
    else:
        stream = FileStream(data_source, allow_nan=True, n_targets=1, target_idx=target_column)
    return stream

def genericDetectorSteps(_detector, data_source, target_column, max_samples, batch_size):
    """
    This function include the general steps for an algorithm to detect anomaly for streaming data.
    predict_and_update() will be different based on the class type.
    :param _detector: The detected algorithm we are going to use
    :param data_source: The dataset link 
    :param batch_size: size of data to send to algorithm to each iteration for anomaly detection
    :param max_samples: maximum number of samples sent
    :param timestamp_column: column name in the data source which represents timestamp
    :param target_column: column name in the data source which represents target column
    :return: detector object
    """    
    _detector.stream = get_file_stream(data_source, target_column)
    _detector.start_time = time.time()

    if _detector.isPretrain:
        _detector.pre_train(_detector.isSupervised)
    else:
        # if not pretrain, skip these data
        _detector.get_next_stream(_detector.initial_batch_training_size)

    while _detector.n_samples < max_samples and _detector.stream.has_more_samples():
        # For LSTM, we need to get next sequence instead of data
        if _detector.name.startswith('AE-LSTM'):
            X, y = _detector.get_next_sequence(batch_size)
        else:
            X, y = _detector.get_next_stream(batch_size)

        _detector.predict_and_update(X,y)

    _detector.end_time = time.time()
    return _detector
       

def get_parser():
    '''
    Get command line input arguments
    :return: parser object containing all command line data
    '''
    # Get parser for command line arguments.
    parser = argparse.ArgumentParser(description="Anamoly Detection in Data Streams",
                                     usage="\n * example usage:  -dataSource ./datasets/sensor.csv -timestampColumn timestamp -timestampFormat '%d-%m-%Y %H:%M' -targetColumn machine_status -batchSize 5 -maxSamples 500 -timesteps_for_LSTM 5\n * ")
    parser.add_argument("-dataSource",
                        dest="data_source",
                        help="String providing complete path for the dataset if local, or kafka link for live data  ")
    parser.add_argument("-timestampColumn",
                        dest="timestamp_column",
                        default=None,
                        help="name of the column havig timestamp values")
    parser.add_argument("-timestampFormat",
                        dest="timestamp_format",
                        default=None,
                        help="format of the column having timestamp values")

    parser.add_argument("-targetColumn",
                        dest="target_column",
                        default=None,
                        help="name of the column having target values, default is None for clustering problem")
    parser.add_argument("-initialBatchTrainingSize",
                        dest="initial_batch_training_size",
                        default="100",
                        help="A integer representing initial training length batch size, default is 100 means uses firt 100 records for initial training.")
    parser.add_argument("-batchSize",
                        dest="batch_size",
                        default="1",
                        help="A integer representing batch size, used to retrieve batch size number of records in every iteration, defualt is 1 .")
    parser.add_argument("-maxSamples",
                        dest="max_samples",
                        default="500",
                        help="A integer representing max samples in the buffer,  defualt is 500.")
    parser.add_argument("-timesteps_for_LSTM",
                    dest="timesteps_for_LSTM",
                    default="5",
                    help="timesteps for LSTM")

    return parser

if __name__ == '__main__':
    """
    Main block responsible for
    1: Parsing user's command line arguments.
    2: Cleaning the data
    3: Based on target column (Dataset with output label and without output label), execute the unsupervised algorithms.
    """

    print("Reading and Analysing Command Line Arguments")
    try:
        parser = get_parser()
        args = parser.parse_args()
    except:
        print("\nTry *** python K-CR.py -h or --help *** for more details.")
        exit(0)

    print("Converting dataset to desired format")
    # Convert Data Source timestamp column to seconds
    data_source = args.data_source
    OUTPUT_FILE_NAME = data_source.split('/')[-1][:-4] + str('_result.csv')
    timestamp_column = args.timestamp_column
    timetamp_format = args.timestamp_format
    target_column = args.target_column
    initial_batch_training_size = int(args.initial_batch_training_size)
    batch_size = int(args.batch_size)
    max_samples = int(args.max_samples)
    timesteps_lstm = int(args.timesteps_for_LSTM)
    target_column_idx, timestamp_column_idx = None, None

    df = pd.read_csv(data_source)
    df = df.fillna(df.mean())
    # drop id if exist
    df = df.drop(['id'], axis=1, errors='ignore')
    original_cols = df.columns.tolist()
    assert TIME_SERIES_DRAW_RIGHT_IDX < len(original_cols)

    if (data_source == './datasets/sensor.csv'):
        df = df.replace('NORMAL', 0)
        df = df.replace('BROKEN', 1)
        df = df.replace('RECOVERING', 1)
    
    cols_to_drop = []
    if(timestamp_column != None):
        cols_to_drop.append(timestamp_column)
        df.dropna(subset=[timestamp_column], inplace=True)
        df[timestamp_column] = df[timestamp_column].apply(lambda x: datetime.timestamp(datetime.strptime(x, timetamp_format)))
        timestamp_column_idx = df.columns.get_loc(timestamp_column)

    if target_column != None:
        le = preprocessing.LabelEncoder()
        cols_to_drop.append(target_column)
        df[target_column] = le.fit_transform(df[target_column])
        target_column_idx = df.columns.get_loc(target_column)
    if (TIME_SERIES_DRAW_LEFT_IDX == timestamp_column) or (TIME_SERIES_DRAW_RIGHT_IDX == target_column):
        print("Please do not enter timestamp or target column for time series plot. ")
        exit(0)
    cols_to_draw = original_cols[TIME_SERIES_DRAW_LEFT_IDX:(TIME_SERIES_DRAW_RIGHT_IDX + 1)]
    column_names = df.drop(cols_to_drop,axis=1).columns.tolist()

    new_data_source = './data_results/'+ data_source.split('/')[-1][:-4]+ "_converted.csv"
    df.to_csv(new_data_source, index=False)

    # Start to train models
    hs_tree = HSTree(name = 'Half-space Tree',detector = HalfSpaceTrees(), \
                            data_source=new_data_source, initial_batch_training_size=initial_batch_training_size, \
                            batch_size=batch_size, max_samples=max_samples, timestamp_column=timestamp_column_idx, target_column=target_column_idx, \
                            isSupervised=False, ifSaveResult = True, isPretrain = False)

    isolate_forest = IsolateForest(name = 'Isolation-Forest Stream', detector = IsolationForestStream(), \
                            data_source=new_data_source, initial_batch_training_size=initial_batch_training_size, \
                            batch_size=batch_size, max_samples=max_samples, timestamp_column=timestamp_column_idx, target_column=target_column_idx, \
                            isSupervised=True, ifSaveResult = True, isPretrain = True)
    isolate_forest_PA = IsolateForest(name = 'Isolation-Forest Stream - PADWIN', detector = PADWINIsolationForestStream(), \
                            data_source=new_data_source, initial_batch_training_size=initial_batch_training_size, \
                            batch_size=batch_size, max_samples=max_samples, timestamp_column=timestamp_column_idx, target_column=target_column_idx, \
                            isSupervised=True, ifSaveResult = True, isPretrain = True)
    
    LOF_model = LOFOnline(n_neighbors=LOF_N_NEIGHBORS, contamination=LOF_CONTAMINATION, name = 'LOF', detector = LocalOutlierFactor(n_neighbors=LOF_N_NEIGHBORS, contamination=LOF_CONTAMINATION), \
                    data_source=new_data_source, initial_batch_training_size=initial_batch_training_size, \
                    batch_size=batch_size, max_samples=max_samples, timestamp_column=timestamp_column_idx, target_column=target_column_idx, \
                    isSupervised=False, ifSaveResult = True, isPretrain = True)
    
    LSTM_model = LSTMOnline(name = 'AE-LSTM', detector = AELSTMOnline(time_steps = timesteps_lstm, neurons = N_NEURONS, epochs=N_EPOCHS, anomaly_rate=30, n_batch=batch_size), \
            data_source=new_data_source, initial_batch_training_size=initial_batch_training_size, \
            batch_size=batch_size, max_samples=max_samples, timestamp_column=timestamp_column_idx, target_column=target_column_idx, \
            isSupervised=False, ifSaveResult = True, isPretrain = True)
    knn_online = OnlinePySad(name = 'KNN',detector = None, data_source=new_data_source, initial_batch_training_size=initial_batch_training_size, \
                            batch_size=batch_size, max_samples=max_samples, timestamp_column=timestamp_column_idx, target_column=target_column_idx, \
                            isSupervised=False, ifSaveResult = True, isPretrain = True)
    
    # More models to add...
    # detectors = [knn_online, isolate_forest_PA, LSTM_model, isolate_forest]
    detectors = [LSTM_model]
    trained_detectors = []
    for m in detectors:
        try:
            trained_m = genericDetectorSteps(m, new_data_source, target_column_idx, max_samples, batch_size)
            trained_detectors.append(trained_m)
            if m.name.startswith('AE-LSTM'):
                K.clear_session()
        except:
            print(f'{m.name} is not working')

    Ensemble_model = EnsembleOnline(detector_list=trained_detectors, name = 'EnsembleOnline', detector =None, \
            data_source=new_data_source, initial_batch_training_size=initial_batch_training_size, \
            batch_size=batch_size, max_samples=max_samples, timestamp_column=timestamp_column_idx, target_column=target_column_idx, \
            isSupervised=False, ifSaveResult = True, isPretrain = True)

    y_voted = Ensemble_model.get_voted_y_pred()
    
    for m in trained_detectors:
        print_result(m, m.name, y_voted=y_voted)
        m.save_results(df)
        #  Draw prediction
        if SHOW_PCA:
            m.draw_PCA(n_comp = PCA_COMP, feature_cols = column_names, showPlot=SHOW_PLOT, draw_y_true=False,y_col_name = target_column)

    if target_column != None:
        # set y true for calculating the accuracy 
        Ensemble_model.set_y_true(trained_detectors[0])
        print_result(Ensemble_model, Ensemble_model.name, y_voted)
    else:
        Ensemble_model.y_true = y_voted
        target_column = 'y_voted'

    Ensemble_model.save_results(df)
    if SHOW_PCA:
        Ensemble_model.draw_PCA(n_comp = PCA_COMP, feature_cols = column_names, showPlot=SHOW_PLOT, draw_y_true=False, y_col_name = target_column)
        Ensemble_model.draw_PCA(n_comp = PCA_COMP, feature_cols = column_names, showPlot=SHOW_PLOT, draw_y_true=True, y_col_name = target_column)
    if SHOW_TIME_SERIES:
        for f in cols_to_draw:
            for m in trained_detectors:
                m.draw_time_series(timestamp_column, col_name=f, num_of_xlabel = 10, draw_y_true=False, y_col_name = target_column, showPlot=SHOW_PLOT)
            Ensemble_model.draw_time_series(timestamp_column, col_name=f, num_of_xlabel = 10, draw_y_true=True, y_col_name = target_column, showPlot=SHOW_PLOT)
    pd.DataFrame(results_to_saved).to_csv('./data_results/' + OUTPUT_FILE_NAME)

