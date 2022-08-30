import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import confusion_matrix
import abc
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import seaborn as sns
from datetime import datetime
from matplotlib.pyplot import figure, show
import matplotlib.dates as mdates
import mplcursors
from pysad.transform.postprocessing import RunningZScorePostprocessor,ZScorePostprocessor
from pysad.models import KNNCAD, RobustRandomCutForest, ExactStorm,LODA, StandardAbsoluteDeviation, KitNet
from pysad.transform.ensemble import AverageScoreEnsembler
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()
OUTPUT_DIR = './data_results/'
class onlineDetector:
    def __init__(self,name, detector,data_source,stream = None,initial_batch_training_size = 250, batch_size = 2, max_samples = 50000, timestamp_column = None, target_column = -1, \
                isSupervised = False, ifSaveResult=False, isPretrain=False):
        self.name = name
        self.detector = detector
        self.data_source = data_source
        self.stream = stream
        self.initial_batch_training_size = initial_batch_training_size
        self.batch_size = batch_size
        self.max_samples = max_samples
        self.timestamp_column = timestamp_column
        self.target_column = target_column
        self.time_stamp = []
        self.n_samples = 0
        self.original_data = []
        self.y_true = []
        self.y_pred = []
        self.len_of_data = 0
        self.start_time = None
        self.end_time = None
        self.isSupervised = isSupervised
        self.ifSaveResult = ifSaveResult
        self.isPretrain = isPretrain
        self.outputFileName = None
        self.hasTrained = False
        # self.score_ = []
    
    def get_next_stream(self, bs):
        X, y= self.stream.next_sample(batch_size=bs)
        if(self.timestamp_column != None):
            self.time_stamp += list(X[:,self.timestamp_column])
            X = np.delete(X, [self.timestamp_column], axis = 1)
        # return X[:,1:], y
        return X, y
        
    def pre_train(self, isSupervised):
        X, y = self.get_next_stream(self.initial_batch_training_size)
        # Using y or not for initial fitting
        if isSupervised:
            # IsolationForestStream must need y to train the model 
            if self.target_column != None:
                self.detector.fit(X, y)
                self.hasTrained = True
        else:
            self.detector.fit(X)
            self.hasTrained = True
    
    @abc.abstractmethod
    def predict_and_update(self, X, y):
        raise NotImplementedError("Please Implement predict_and_update")

    def get_num_of_anomaly(self):
        # Assume 1 is Anomaly, 0 is normal
        return np.array(self.y_pred).sum()
    
    def outputFileNameSetter(self, fileFormat):
        file_name = self.data_source.split('/')[-1][:-4]
        file_name = OUTPUT_DIR + file_name
        self.outputFileName =  f'{file_name}_{self.name.replace(" ", "_")}.{fileFormat}'
        return self.outputFileName

    def results(self):
        """
        Funtion to print all requred metrics
        :param y_true: Orignal value of the output variable
        :param y_pred: Predicted value of the output variable
        :return: None, prints the metrics like accuracy, AUC, confusion matrix and classification report
        """
        target_names = ['normal', 'anamoly']
        print("Accuarcy: " + str(accuracy_score(self.y_true, self.y_pred)))
        print("AUC: " + str(roc_auc_score(self.y_true, self.y_pred)))
        print("AUC PR: " + str(average_precision_score(self.y_true, self.y_pred)))
        print(confusion_matrix(self.y_true, self.y_pred))
        print(classification_report(self.y_true, self.y_pred, target_names=target_names, digits=4))
        return {'name': self.name, \
                'Accuarcy': str(accuracy_score(self.y_true, self.y_pred)), \
                'AUC': str(roc_auc_score(self.y_true, self.y_pred)), \
                'AUC PR': str(average_precision_score(self.y_true, self.y_pred)), \
                'confusion matrix': confusion_matrix(self.y_true, self.y_pred), \
                'report': (classification_report(self.y_true, self.y_pred, target_names=target_names, digits=4, output_dict=True))}
    
    def save_results(self, df):
        if len(self.y_pred) == 0:
            return
        df = pd.read_csv(self.data_source)
        df_num_of_row = df.shape[0] 
        # remove pre-train data since it has no predictions
        df = df.iloc[self.initial_batch_training_size:,:].reset_index(drop=True)
        if df_num_of_row > len(self.y_pred):
            df = df.iloc[0:len(self.y_pred),:].reset_index(drop=True)
        df['prediction'] = self.y_pred
        # saving y voted if dataset is unlabelled
        if (self.target_column == None):
            df['y_voted'] = self.y_true
        outs_file = self.outputFileNameSetter('csv')
        df.to_csv(outs_file, index=False)
        print(f"Output data is added with  {self.name} predicted values and data file named as: {self.outputFileName}")

    def draw_PCA(self, n_comp = 3, feature_cols = [], showPlot=False, draw_y_true=False, y_col_name = None):
        # https://towardsdatascience.com/anomaly-detection-with-isolation-forest-visualization-23cd75c281e2
        # plt.figure()
        ifSaveLoss = self.name.startswith('AE-LSTM') & (y_col_name != 'y_voted') & (y_col_name != None)
        if ((draw_y_true) & (y_col_name == None)):
            # print("Unable to plot the true graph because of the missing true y label")
            return
        df = pd.read_csv(self.outputFileName)
        dot_color = 'blue'

        titleName = self.name
        if draw_y_true:
            outliers = df.loc[df[y_col_name] == 1]
            titleName += "_True"
            dot_color = 'green'
        else:
            outliers = df.loc[df['prediction'] == 1]
            titleName += "_Prediction"

        outlier_index = list(outliers.index)
        normal_index = list(set(df.index) - set(outlier_index))
        if n_comp == 2:
            self.draw_PCA_dim_2(df[feature_cols], titleName, normal_index, showPlot, draw_y_true)
            if ifSaveLoss:
                self.save_lstm_loss(df, normal_index, outlier_index, True, y_col_name)
                self.save_lstm_loss(df, normal_index, outlier_index, False, y_col_name)
            return 
        pca = PCA(n_components=n_comp)
        scaler = StandardScaler()
        #normalize the metrics
        X = scaler.fit_transform(df[feature_cols])
        X_reduce = pca.fit_transform(X)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_zlabel("x_composite_3")
        # Plot the compressed data points
        ax.scatter(X_reduce[normal_index, 0], X_reduce[normal_index, 1], zs=X_reduce[normal_index, 2], s=4, lw=1, label="inliers",c=dot_color)
        # Plot x's for the ground truth outliers
        ax.scatter(X_reduce[outlier_index,0],X_reduce[outlier_index,1], X_reduce[outlier_index,2],
                lw=2, s=60, marker="x", c="red", label="outliers")
        ax.legend()
        plt.title(titleName)
        mplcursors.cursor(hover=True)
        plt.savefig(f'./figures/{titleName}.png')
        if showPlot:
            plt.show()
        if ifSaveLoss:
            self.save_lstm_loss(df, normal_index, outlier_index, False, y_col_name)
            self.save_lstm_loss(df, normal_index, outlier_index, True, y_col_name)

    def save_lstm_loss(self, df, normal_index, outlier_index, draw_y_true, y_col_name):
        # plt.figure()
        if y_col_name == None:
            return
        df['loss'] = self.detector.loss
        file_name = 'Loss_' + self.name
        if draw_y_true:
            outliers = df.loc[df[y_col_name] == 1]
            file_name += '  True'
            outlier_index = list(outliers.index)
            normal_index = list(set(df.index) - set(outlier_index))
        else:
            file_name += '  Prediction'
        outs = df.iloc[outlier_index,:]
        ins = df.iloc[normal_index,:]
        plt.title(file_name)
        ax = plt.gca()
        ax.scatter(outs.index, outs['loss'], s=30, c="red", label="outliers", edgecolors='white', alpha=0.8)
        ax.scatter(ins.index, ins['loss'], s=30, c="blue", label="outliers",  alpha=0.8,  linestyle='None')
        # plt.show()
        plt.savefig(f'./figures/{file_name}.png')
        # clean the figure since we will not show the image
        plt.clf()



    def draw_PCA_dim_2(self, df, name, normal_index, showPlot, draw_y_true):
        normal_color = 'green' if draw_y_true else 'blue'
        sns.set_style("darkgrid")
        pca = PCA(2)
        res=pd.DataFrame(pca.fit_transform(df), columns = ['x1','x2'])
        res['anomaly_class'] = np.where(df.index.isin(normal_index), 'normal', 'outlier')
        normal_df = res[res['anomaly_class'] == 'normal']
        outlier_df = res[res['anomaly_class'] == 'outlier']
        plt.title(name)
        # sns.scatterplot(data=res, x='x1', y='x2', hue='anomaly_class', palette = {'normal':normal_color, 'outlier':'red'}, hue_order=['normal', 'outlier'], label={'normal':'normal', 'outlier':'outlier'})
        ax = plt.gca()
        ax.scatter(normal_df['x1'], normal_df['x2'], s=30, c=normal_color, label="normal", edgecolors='white', alpha=0.8)
        ax.scatter(outlier_df['x1'], outlier_df['x2'], s=30, c="red", label="outliers", edgecolors='white', alpha=0.8)
        ax.legend()
        mplcursors.cursor(hover=True)

        if draw_y_true:
            plt.savefig(f'./figures/{self.name}_true_2dim.png')
        else:
            plt.savefig(f'./figures/{self.name}_prediction_2dim.png')
        if showPlot:
            plt.show()

    def draw_time_series(self, timestamp_column, col_name = 'f1', num_of_xlabel = 10, draw_y_true=False, y_col_name = None, showPlot=True):
        # if we need to draw true y but we do not have y columns --> return 
        if draw_y_true & (y_col_name == None):
            print("Missing y value, so not able to draw the true value")
            return
        # if we do not have timestamp column, we cannot draw it. 
        if timestamp_column == None:
            return
        line_color = 'blue'
        if draw_y_true:
            line_color = 'olive'
        df = pd.read_csv(self.outputFileName)
        df['date'] = df[timestamp_column].apply(lambda x: datetime.fromtimestamp(int(x)))
        df['datetime'] = df['date'].apply(lambda x: x.strftime('%Y-%m-%d %H:%M'))
        # f,a = plt.subplots(figsize=(9, 9))
        figure(figsize=(9, 9))
        plt.plot( 'datetime', col_name, data=df, marker='', color=line_color, linewidth=1.3, label='normal')
        plt.xticks(rotation=25)
        ax = plt.gca()
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        
        titleName = self.name
        if draw_y_true & (y_col_name != None):
            outliers = df.loc[df[y_col_name] == 1]
            titleName += f"_{col_name}_Time_True"
        else:
            outliers = df.loc[df['prediction'] == 1]
            titleName += f"_{col_name}_Time_Prediction"

        outlier_index = list(outliers.index)
        plt.title(f'{titleName}: Actual {col_name} values with outliers in red')
        ax.scatter(outlier_index, df.iloc[outlier_index][col_name], s=25, c="red", label="outliers", alpha=0.8)
        ax.legend()
        mplcursors.cursor(hover=True)
        plt.savefig(f'./img/{titleName}.png')
        if showPlot:
            plt.show()
    
    

class HSTree(onlineDetector):
    
    def predict_and_update(self, X, y):
        # saving the original copy of data for plotting
        # self.original_data += list(X)
        y_pred = self.detector.predict(X)
        # X = X[y_pred == 0]
        self.detector.partial_fit(X, y_pred)
        # Update the number of data
        self.n_samples += len(y_pred)
        self.len_of_data += len(y_pred)

        self.y_pred += list(y_pred)
        self.y_true += list(y)

class IsolateForest(onlineDetector):
    
    def predict_and_update(self, X, y):
        # saving the original copy of data for plotting
        # self.original_data += list(X)
        temp_y_pred = []
        for data in X:
            y_pred = self.detector.predict([data])
            # isolation forest will return -1 if it is the last element        
            if y_pred != [1]:
                y_pred = [0]
                
            temp_y_pred.append(y_pred[0])

        self.y_pred += list(temp_y_pred)
        # print(self.y_pred)
        self.detector.partial_fit(X, temp_y_pred)
        # # Update the number of data
        self.n_samples += len(temp_y_pred)
        self.len_of_data += len(temp_y_pred)
        self.y_true += list(y)

class OnlineKCR(onlineDetector):

    def __init__(self, cluster, threshold, **kwargs):
        super(OnlineKCR, self).__init__( **kwargs)
        self.cluster = cluster
        self.threshold = threshold

    def predict_and_update(self, X, y):
        y_result = self.detector.predict(X, inside=1)
        for k in range(len(y_result)):
            self.detector.partial_fit([X[k]], y_result[k])
        for z in y_result:
            if (int(z) == (self.cluster + 1)):
                self.y_pred.append(1)
            else:
                self.y_pred.append(0)

        # Update the number of data
        self.n_samples += len(y_result)
        self.len_of_data += len(y_result)
        self.y_true += list(y)

class LOFOnline(onlineDetector):

    def __init__(self, n_neighbors, contamination, **kwargs):
        super(LOFOnline, self).__init__( **kwargs)
        self.n_neighbors = n_neighbors
        self.contamination = contamination

    def predict_and_update(self, X, y):
        # saving the original copy of data for plotting
        # self.original_data += list(X)
        y_result = self.detector.fit_predict(X)
        for k in y_result:
            if k == -1:
                self.y_pred.append(1)
            else:
                self.y_pred.append(0)

        # Update the number of data
        self.n_samples += len(y_result)
        self.len_of_data += len(y_result)
        self.y_true += list(y)

class EnsembleOnline(onlineDetector):
    def __init__(self, detector_list, **kwargs):
        super(EnsembleOnline, self).__init__( **kwargs)
        self.detector_list = detector_list

    def get_voted_y_pred(self):
        models = {}
        # Ensemble method is always trained -> use the voted y values
        self.hasTrained = True
        for m in self.detector_list:
            models[m.name] = m.y_pred

        df = pd.DataFrame(models)
        self.len_of_data = df.shape[0]
        df['normal'] = (df == 0).astype(int).sum(axis=1)
        df['anomaly'] = (df.drop(['normal'], axis=1) == 1).astype(int).sum(axis=1)
        df['y_voted'] = np.where(df.anomaly >= df.normal, 1, 0)
        df.to_csv("./data_results/y_voted.csv", index=False)
        self.y_pred = df['y_voted'].tolist()
        return self.y_pred

    def set_y_true(self,m):
        self.y_true = m.y_true

class LSTMOnline(onlineDetector):
    sliding_X = []
    n_features = 0

    def init_sequences(self,X, time_steps):
        assert (self.initial_batch_training_size > time_steps), "Please enter a lager initial batch size for training"
        Xs = []
        for i in range(len(X) - time_steps + 1):
            v = X[i:(i + time_steps),:]
            Xs.append(v)
        Xs = np.asarray(Xs).astype(np.float32)
        self.sliding_X = Xs[-1].reshape(1,time_steps,self.n_features)
        return Xs
    
    def get_next_sequence(self, batch_size):
        assert len(self.sliding_X) > 0, 'Please pretrain the LSTM'
        Xs = []
        X, y = self.get_next_stream(batch_size)
        self.detector.XForKCR = X
        for x in X:
            # pop the oldest data
            self.sliding_X = self.sliding_X[:,1:,:]
            # Parital update the mean,std for scaler
            self.detector.global_scaler.partial_fit(x.reshape(1, -1))
            # Tranfrom X
            x = self.detector.global_scaler.transform(x.reshape(1, -1))
            # Append the new data X into slding window
            self.sliding_X = np.insert(self.sliding_X, self.detector.time_steps-1, x, axis=1)
            Xs.append(self.sliding_X[0])

        return np.array(Xs), y

    def pre_train(self, isSupervised):
        X, _ = self.get_next_stream(self.initial_batch_training_size)
        self.detector.XForKCR = np.array(X)
        self.detector.global_scaler.fit(X)
        X = self.detector.global_scaler.transform(X)
        # self.detector.XForKCR = np.array(X)
        self.n_features = X.shape[-1]

        X_train = self.init_sequences(X,self.detector.time_steps)
        self.detector.create_AELSTM(X_train)
        self.hasTrained = True
        # calculate tne mean and std for training dataset
        self.detector.fit(X_train)


    def predict_and_update(self, X, y):
        # X come in with dim [sample, timesteps, n_features]
        preds = list(self.detector.fit_predict(X,y))
        self.y_pred += preds
        # Update the number of data
        self.n_samples += len(preds)
        self.len_of_data += len(preds)
        self.y_true += list(y)

class OnlinePySad(onlineDetector):
    std_scaler = StandardScaler()
    score_scaler = RunningZScorePostprocessor(window_size=5)
    random_cut = RobustRandomCutForest(num_trees=10, shingle_size=4, tree_size=256)
    exact_storm = ExactStorm(window_size=500, max_radius=0.1)
    loda_model = LODA(num_bins=10, num_random_cuts=100)
    knn = KNNCAD(probationary_period=100)
    minmax = MinMaxScaler()
    pysad_models = [random_cut]
    score_ = []
    ensembler = AverageScoreEnsembler()
    def pre_train(self, isSupervised):
        X, _ = self.get_next_stream(self.initial_batch_training_size)
        self.hasTrained = True
        for m in self.pysad_models:
            m.fit(X)

    def predict_and_update(self, X, y):
        model_scores = np.empty(len(self.pysad_models), dtype=np.float)
        y_result = []
        # Fit & Score via for each model.
        for x in X:
            model_scores = np.empty(len(self.pysad_models), dtype=np.float)

            for i, model in enumerate(self.pysad_models):
                model.fit_partial(x)
                model_scores[i] = model.score_partial(x)
            score = self.ensembler.fit_transform_partial(model_scores) 
            y_result.append(score)
        y_result = np.array(y_result)
        y_result = np.ravel(y_result)
        self.score_ += list(y_result)
        y_result = np.where(y_result > 15, 1, 0)

        self.y_pred += list(y_result)
        # Update the number of data
        self.n_samples += len(y_result)
        self.len_of_data += len(y_result)
        self.y_true += list(y)

# This is the main function only used for testing the class
if __name__ == "__main__":
    pass