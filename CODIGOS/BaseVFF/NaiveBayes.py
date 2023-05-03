import time
import os
import tensorflow as tf
from keras import backend as K
import numpy as np
from PIL import Image
import pandas as pd
import glob
import matplotlib.pyplot as plt
import csv
# from sklearn.model_selection import GridSearchCV
from tune_sklearn import TuneSearchCV
from tune_sklearn import TuneGridSearchCV
from sklearn.linear_model import SGDClassifier
from ray import tune

from sklearn.naive_bayes import GaussianNB
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingRandomSearchCV
from skimage.feature import local_binary_pattern
from skimage.color import rgb2gray
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
#pip install imbalanced-learn
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
from sklearn import preprocessing
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
import subprocess
from ray.tune.suggest.bayesopt import BayesOptSearch
from xgboost import XGBClassifier


from sklearn.metrics import auc, roc_curve, roc_auc_score, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import datetime


# tf.config.list_physical_devices('GPU')

# import os
# os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
# os.environ['CUDA_VISIBLE_DEVICES'] = "0"

##Variaveis globais
save_metrics_path = "../Metrics/"
save_net_name_test = "bayes_test.csv"
save_net_name_test2 = "bayes_testKeras.csv"
save_net_name_train = "bayes_train.csv"
base_path_parts = "../PartitionsFeatures/"
files_parts = os.listdir(base_path_parts)
input_size = (80,80)
runtimeTrain = 0.0
runtimeTest = 0.0



def specificity(tn, fp):
    return tn / (tn + fp)

# Negative Predictive Error
def npv(tn, fn):
    return tn / (tn + fn + 1e-7)

# Matthews Correlation_Coefficient
def mcc(tp, tn, fp, fn):
    num = tp * tn - fp * fn
    den = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    return num / np.sqrt(den + 1e-7)



def calculateMeasuresTest(y_pred, y_true, scores, folder):
    metricsTest = pd.DataFrame()
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    #fpr, tpr, _ = roc_curve(y_true, scores, pos_label=2)
    auc_val = roc_auc_score(y_true, scores)

    # Test RESULTS
    metricsTest['folder'] = [folder]
    metricsTest['accuracy'] = [accuracy_score(y_true, y_pred)]
    metricsTest['precision'] = [precision_score(y_true, y_pred)]
    metricsTest['sensitivity'] = [recall_score(y_true, y_pred)]
    metricsTest['specificity'] = [specificity(tn,fp)]
    metricsTest['fmeasure'] = [f1_score(y_true, y_pred)]
    metricsTest['npv'] = [npv(tn, fn)]
    metricsTest['mcc'] = [mcc(tp, tn, fp, fn)]
    metricsTest['auc'] = [auc_val]
    metricsTest['tn'] = [tn]
    metricsTest['fp'] = [fp]
    metricsTest['fn'] = [fn]
    metricsTest['tp'] = [tp]
    metricsTest['runtime'] = [runtimeTest]

    print(metricsTest)

    if os.path.exists(os.path.join(save_metrics_path, save_net_name_test)):
        metricsTest.to_csv(os.path.join(save_metrics_path, save_net_name_test), sep=',', mode='a', index=False, header=False)
    else:
        metricsTest.to_csv(os.path.join(save_metrics_path, save_net_name_test), sep=',', index=False)

   
#Funções importantes
def calculateMeasuresTrain(y_pred, y_true, scores, folder):
    metricsTrain = pd.DataFrame()
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    #fpr, tpr, _ = roc_curve(y_true, scores, pos_label=2)
    auc_val = roc_auc_score(y_true, scores)

    # TRAIN RESULTS
    metricsTrain['folder'] = [folder]
    metricsTrain['accuracy'] = [accuracy_score(y_true, y_pred)]
    metricsTrain['precision'] = [precision_score(y_true, y_pred)]
    metricsTrain['sensitivity'] = [recall_score(y_true, y_pred)]
    metricsTrain['specificity'] = [specificity(tn,fp)]
    metricsTrain['fmeasure'] = [f1_score(y_true, y_pred)]
    metricsTrain['npv'] = [npv(tn, fn)]
    metricsTrain['mcc'] = [mcc(tp, tn, fp, fn)]
    metricsTrain['auc'] = [auc_val]
    metricsTrain['tn'] = [tn]
    metricsTrain['fp'] = [fp]
    metricsTrain['fn'] = [fn]
    metricsTrain['tp'] = [tp]
    metricsTrain['runtime'] = [runtimeTrain]

    print(metricsTrain)

    if os.path.exists(os.path.join(save_metrics_path, save_net_name_train)):
        metricsTrain.to_csv(os.path.join(save_metrics_path, save_net_name_train), sep=',', mode='a', index=False, header=False)
    else:
        metricsTrain.to_csv(os.path.join(save_metrics_path, save_net_name_train), sep=',', index=False)   


def load_dataset(base_path):
    imagens, labels = list(), list()
    classes = os.listdir(base_path)
    for c in classes:
        for p in glob.glob(os.path.join(base_path, c, '.csv')):
            imagens.append(p)
            labels.append(c)
    
    return np.asarray(imagens), labels

def load_dataset_part(step):
    train_Y, test_y = list(), list()

    trainFrac = pd.io.parsers.read_csv(os.path.join(base_path_parts, "%.2d-train-fractures.csv"%(step)), skiprows=1)
    for i in range(0,len(trainFrac), 1):
        train_Y.append('ComFratura')
    trainNoFrac = pd.io.parsers.read_csv(os.path.join(base_path_parts, "%.2d-train-nofractures.csv"%(step)), skiprows=1)
    for i in range(0,len(trainNoFrac), 1):
        train_Y.append('SemFratura')

    train_X = np.concatenate((trainFrac, trainNoFrac), axis=0)

    testFrac = pd.io.parsers.read_csv(os.path.join(base_path_parts, "%.2d-test-fractures.csv"%(step)), skiprows=1)
    for i in range(0,len(testFrac), 1):
        test_y.append('ComFratura')
    testNoFrac = pd.io.parsers.read_csv(os.path.join(base_path_parts, "%.2d-test-nofractures.csv"%(step)), skiprows=1)
    for i in range(0,len(testNoFrac), 1):
        test_y.append('SemFratura')

    test_x = np.concatenate((testFrac, testNoFrac), axis=0)
    
    return train_X, test_x, train_Y, test_y

def load_balance_class_parts(step):
    train_X, test_x, train_Y, test_y = load_dataset_part(step)
    
    ##Balanceameto dos dados de treino
    undersample = RandomUnderSampler(sampling_strategy='majority')
    train_under_X, train_under_Y = undersample.fit_resample(train_X, train_Y)
    # print(train_under_Y)
    
    lb = LabelBinarizer()
    min_max_scaler = preprocessing.MinMaxScaler()
    # train_under_X = min_max_scaler.fit_transform(train_under_X)
    train_under_Y = lb.fit_transform(train_under_Y)
    
    ##Balanceamento dos dados de test
    undersample = RandomUnderSampler(sampling_strategy='majority')
    test_under_X, test_under_Y = undersample.fit_resample(test_x, test_y)
    
    lb = LabelBinarizer()
    min_max_scaler = preprocessing.MinMaxScaler()
    test_under_Y = lb.fit_transform(test_under_Y)
    # test_under_X = min_max_scaler.fit_transform(test_under_X)
    
    return train_under_X, train_under_Y, test_under_X, test_under_Y 



def train_ml_algorithm(X_train, y_train):
    clf = GaussianNB()
    parameter_grid = {
        'var_smoothing': np.logspace(0,-9, num=1000),
    }

    tune_search = GridSearchCV(clf,
        parameter_grid,n_jobs=-1, verbose=1, cv=2)

    seach_classfier = tune_search.fit(X_train, y_train.ravel())

    return seach_classfier

def save_parts_proc(part):
    with open(os.path.join(save_metrics_path, "bayes/parts.txt"), mode="a") as f:
        f.write(f"{part}\n")
        
def load_parts_proc():
    parts = []
    with open(os.path.join(save_metrics_path, "bayes/parts.txt"), mode="r") as f:
        parts = f.readlines()

    parts = [p.replace("\n", "") for p in parts]
    
    return parts

if __name__ == '__main__':
    #metrics = pd.DataFrame()
    
    #parts = load_parts_proc()
    
    for step in range(1,101,1):
        # if folder in parts:
        #     continue

        print(f"Step: {step}")
        
        print("Load features")
        X_feat_train, train_under_Y, X_feat_test, test_under_Y = load_balance_class_parts(step)
    
        
        print("Trainning Bayes")
        start_train = time.time()
        
        initClassifier = train_ml_algorithm(X_feat_train, train_under_Y)
        y_pred_train  = initClassifier.predict(X_feat_train)
        y_proba_train = initClassifier.predict_proba(X_feat_train)[:, 1]
        
        runtimeTrain = time.time() - start_train
        print("Bayes Trained in %2.2f seconds"%(runtimeTrain))


        print("Testing Bayes")
        start_test = time.time()
        y_pred_test = initClassifier.predict(X_feat_test)
        y_proba_test = initClassifier.predict_proba(X_feat_test)[:, 1]
        
        runtimeTest = time.time() - start_test
        print("Bayes Tested in %2.2f seconds"%(runtimeTest)) 
        

        calculateMeasuresTrain(y_pred_train.reshape(y_pred_train.shape[0], 1), train_under_Y, y_proba_train, step)
        calculateMeasuresTest(y_pred_test.reshape(y_pred_test.shape[0], 1), test_under_Y, y_proba_test, step)
        break

