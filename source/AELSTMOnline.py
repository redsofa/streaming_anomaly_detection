# from tensorflow import keras
import enum
from math import exp
from re import L
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, TimeDistributed, RepeatVector
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
import pandas as pd
# from tensorflow import keras
from skmultiflow.drift_detection.adwin import ADWIN
from tensorflow.keras import backend as K
from tdigest import TDigest
import tensorflow as tf
physical_device = tf.config.experimental.list_physical_devices('GPU')
print("Number of GPUs Available: ", len(physical_device))
# use a CUDA GPU it exits
if len(physical_device):
    tf.config.experimental.set_memory_growth(physical_device[0], True)
    
# Global Variables Defined space:
# Adjust MU and percentile to a lower values if the anomaly rate is large.
ROLLING_MEAN_MU = 3.2
ROLLING_WINDOW_SIZE = 150
LSTM_PARTIAL_FIT_LR = 0.0001
PERCENTILE_ = 90


class SlidingWindows:
    def __init__(self, ws=500, anomaly_rate=30):  # ws =250
        self.window_size = ws
        self.windows = []
        self.predictions = []
        self.anomaly_rate = anomaly_rate
        self.change_ws = 0

    def append_data(self, X, y):
        if(len(self.windows) < self.window_size):
            self.windows += list(X)
            self.predictions += list(y)
        else:
            self.pop_first_n(len(X))
            self.windows += list(X)
            self.predictions += list(y)

        self.change_ws = len(self.predictions)

    def clean_drift_preds(self):
        len_of_preds = len(self.predictions)
        self.predictions = [0] * len_of_preds

    def isDrift(self):
        return (np.array(self.predictions).sum() > self.anomaly_rate)
    
    def pop_first_n(self, n):
        self.windows = self.windows[n:]
        self.predictions = self.predictions[n:]
    
    def get_windows(self):
        return np.array(self.windows)
    
    def get_normal_points(self):
        not_anomaly = np.array(self.predictions) != 1
        return self.get_windows()[not_anomaly]
    
    def get_predictions(self):
        return self.predictions

# Rolling mean with MAD
class RollingMean:
    def __init__(self, window_size):
        self.ws = window_size
        self.mean_ = 0
        self.std_ = 0
        self.sliding_window = None
        self.mu = 1.485
    
    def init_rolling_mean(self, X, mu):
        self.sliding_window = np.array(X)
        self.mean_ = np.median(self.sliding_window)
        self.std_ = np.median(np.abs(self.sliding_window - self.mean_ ))
        self.mu = mu

    def update_with_batch(self, Xs):
        for x in Xs:
            self.update(x, False)
        self.mean_ = np.median(self.sliding_window)
        self.std_ = np.median(np.abs(self.sliding_window - self.mean_ ))

    def update(self, new_val, update_val = True):
        if(len(self.sliding_window) < self.ws):
            self.sliding_window = np.append(self.sliding_window, new_val)
        else:
            self.sliding_window = self.sliding_window[1:]
            self.sliding_window = np.insert(self.sliding_window, self.ws-1, new_val)
        if update_val:
            self.mean_ = np.median(self.sliding_window)
            self.std_ = np.median(np.abs(self.sliding_window - self.mean_ ))
    
    def getThreshold(self):
        return np.array(self.mean_ + self.mu * self.std_)
    
    def __repr__(self):
        return 'mean: ' + str(self.mean_) + ' std: ' + str(self.std_)

class AELSTMOnline:
    def __init__(self, time_steps=5, neurons=64, epochs=50, anomaly_rate=30, n_batch=5, ws=500):
        self.time_steps = time_steps
        self.neurons = neurons
        self.exp_anomaly = None
        self.model = None 
        self.current_seq = []
        self.global_scaler = StandardScaler()
        self.epochs = epochs
        self.anomaly_rate = anomaly_rate
        self.verbose = 0
        self.n_batch = n_batch
        self.loss = []
        self.rolling_mean = RollingMean(ROLLING_WINDOW_SIZE)
        self.current_window = SlidingWindows(ws, anomaly_rate)
        self.window_size = ws
        self.isDrift = False
        self.score_ = []
        self.adwin = ADWIN()
        self.name_ = 'AE-LSTM'
        self.use_drift = True
        self.use_adwin = True
        self.percentile_ranker = TDigest()

    def create_AELSTM(self, X):
        time_steps = self.time_steps
        n_features = X.shape[-1]
        bottleneck = int(self.neurons / 2)
        model = Sequential()
        # Encoder
        model.add(LSTM(self.neurons,  input_shape=(time_steps, n_features), return_sequences=True))
        model.add(Dropout(0.3))
        model.add(LSTM(bottleneck,  return_sequences=True))
        model.add(LSTM(int(bottleneck/2),  return_sequences=False, name='latent'))
        model.add(RepeatVector(time_steps))
        # Decoder
        model.add(LSTM(bottleneck, return_sequences=True))
        model.add(LSTM(int(bottleneck/2), return_sequences=True))
        model.add(Dropout(0.3))
        model.add(LSTM(self.neurons,  return_sequences=True))
        # Output dense layer
        model.add(TimeDistributed(Dense(n_features)))
        # print(model.summary())
        model.compile(optimizer='adam', loss='mae')
        print(self.name_ + ' start traning')
        history = model.fit(
        X, X,
        epochs=self.epochs,
        batch_size=32,
        validation_split=0.1,
        shuffle=False, verbose=self.verbose)
        self.model = model
        return model
    
    def calculate_loss(self, X):
        X_pred = self.model(X)
        # train_mae_loss is 2-d array, [sample, n_features]
        last_timesteps = np.abs(X_pred - X)[:,-1,:].reshape(X_pred.shape[0],1,X_pred.shape[2])
        train_mae_loss = np.mean(last_timesteps, axis=1)
        # sum over all the features --> 1-d array [samples,]
        train_mae_loss_avg_vector = np.mean(train_mae_loss, axis=1)
        return np.sqrt(train_mae_loss_avg_vector)

    def fit(self, X):
        train_mae_loss_avg_vector = self.calculate_loss(X)
        if self.name_ == 'AE-LSTM':
            self.rolling_mean.init_rolling_mean(train_mae_loss_avg_vector[-self.rolling_mean.ws:], ROLLING_MEAN_MU)
            self.percentile_ranker = TDigest()
            self.percentile_ranker.batch_update(list(train_mae_loss_avg_vector))

        return train_mae_loss_avg_vector.reshape(-1, 1)

    
    def partial_fit(self, n_epochs, X, bs):
        if X.shape[0] == 0: return
        K.set_value(self.model.optimizer.learning_rate, LSTM_PARTIAL_FIT_LR)
        for _ in range(n_epochs):
                self.model.fit(X, X, epochs=1, batch_size=bs, verbose=self.verbose, shuffle=False)
                # self.model.reset_states()

    def getMovingAvePrediction(self, loss_vector, moving_avg):
        preds = []
        for l in (loss_vector):
            thresholds = np.array(moving_avg.getThreshold())
            if l < thresholds:
                preds.append(0)
            else:
                preds.append(1)
            moving_avg.update(l)
        return np.array(preds)
    
    def get_predictions(self, loss_1d, rolling_preds):
        percentile_cut_off = self.percentile_ranker.percentile(PERCENTILE_)
        ranker_preds = np.where(loss_1d >  percentile_cut_off , 1, 0)
        return ranker_preds & rolling_preds


    def fit_predict(self, X,y):
        train_mae_loss_avg_vector = self.calculate_loss(X)
        loss_list = list(train_mae_loss_avg_vector)
        self.loss += loss_list
        self.percentile_ranker.batch_update(loss_list)
        rolling_preds = self.getMovingAvePrediction(train_mae_loss_avg_vector, self.rolling_mean)
        preds = self.get_predictions(train_mae_loss_avg_vector, rolling_preds)

        if self.use_drift:
            self.current_window.append_data(X, preds)
            for p in preds:
                self.adwin.add_element(p)
                if self.adwin.detected_change():
                    print(self.name_ + ' drift detected...')
                    X_refit = self.current_window.get_windows()
                    # re-create the LSTM model
                    self.create_AELSTM(X_refit)
                    # update the moving thresholds
                    self.fit(X_refit)
                    self.adwin.reset()
                    break
        # only use normal data to fit the model
        not_anomaly = (preds != 1)
        X_normal = X[not_anomaly]
        self.partial_fit(1, X_normal, 5)
        return preds