import time
import os
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
import tensorflow as tf
from keras import backend as K
import numpy as np
from PIL import Image
import pandas as pd
import glob
import matplotlib.pyplot as plt
import csv
import ray
from sklearn.model_selection import ShuffleSplit
from ray import tune
from tune_sklearn import TuneGridSearchCV
# from sklearn.model_selection import GridSearchCV
from tune_sklearn import TuneSearchCV
from tune_sklearn import TuneGridSearchCV
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

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
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier, GradientBoostingClassifier, StackingClassifier, VotingClassifier, HistGradientBoostingClassifier
from sklearn.svm import NuSVC, SVC, LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


from sklearn.metrics import auc, roc_curve, roc_auc_score, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import datetime

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# Runnin: 
# 'SVMLinear', 'SVMNu' ,

# DONE: 'DAnalysisQuadratic', 'GradientBoosting',  'HistGradient', 'ExtraTrees'
methodsNames = [  'NBayes'] 
# 'KNN', 'RandomForest', , , 'DTree', 'NBayes', 'AdaBoost', , 'SVMC'

import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['CUDA_VISIBLE_DEVICES'] = "1"

##Variaveis globais
save_metrics_path = "../Datasets/DatasetBalanced2/Results/csvs/"
base_path_parts = "../Datasets/DatasetBalanced2/Features/"
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


def calculateMeasures(Y_pred, Y_true, Yscores, y_pred, y_true, yscores, folder, methodName, thresh):
    metrics = pd.DataFrame()
    tn, fp, fn, tp = confusion_matrix(Y_true, Y_pred, labels=[0,1]).ravel()
    #fpr, tpr, _ = roc_curve(y_true, scores, pos_label=2)
    auc_val = roc_auc_score(Y_true, Yscores)

    metrics['folder'] = [folder]
    metrics['network'] = [methodName]

    # Train RESULTS
    metrics['accuracy'] = [accuracy_score(Y_true, Y_pred)]
    metrics['precision'] = [precision_score(Y_true, Y_pred)]
    metrics['sensitivity'] = [recall_score(Y_true, Y_pred)]
    metrics['specificity'] = [specificity(tn,fp)]
    metrics['fmeasure'] = [f1_score(Y_true, Y_pred)]
    metrics['npv'] = [npv(tn, fn)]
    metrics['mcc'] = [mcc(tp, tn, fp, fn)]
    metrics['auc'] = [auc_val]
    metrics['tn'] = [tn]
    metrics['fp'] = [fp]
    metrics['fn'] = [fn]
    metrics['tp'] = [tp]
    metrics['runtime'] = [runtimeTrain]

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    #fpr, tpr, _ = roc_curve(y_true, scores, pos_label=2)
    auc_val = roc_auc_score(y_true, yscores)

    # Test RESULTS
    metrics['val_accuracy'] = [accuracy_score(y_true, y_pred)]
    metrics['val_precision'] = [precision_score(y_true, y_pred)]
    metrics['val_sensitivity'] = [recall_score(y_true, y_pred)]
    metrics['val_specificity'] = [specificity(tn,fp)]
    metrics['val_fmeasure'] =[f1_score(y_true, y_pred)]
    metrics['val_npv'] = [npv(tn, fn)]
    metrics['val_mcc'] = [mcc(tp, tn, fp, fn)]
    metrics['val_auc'] = [auc_val]
    metrics['val_tn'] = [tn]
    metrics['val_fp'] = [fp]
    metrics['val_fn'] = [fn]
    metrics['val_tp'] = [tp]
    metrics['val_runtime'] = [runtimeTest]

    print(bcolors.FAIL + 'ACC: %.2f' %(100*metrics['val_accuracy'][0]) + ' AUC: %.2f' %(100*metrics['val_auc'][0]) + bcolors.ENDC)

    if os.path.exists(os.path.join(save_metrics_path, methodName + str(thresh*100) + '.csv')):
        metrics.to_csv(os.path.join(save_metrics_path, methodName + str(thresh*100) + '.csv'), sep=',', mode='a', index=False, header=False)
    else:
        metrics.to_csv(os.path.join(save_metrics_path, methodName + str(thresh*100) + '.csv'), sep=',', index=False)  


def load_dataset(base_path):
    imagens, labels = list(), list()
    classes = os.listdir(base_path)
    for c in classes:
        for p in glob.glob(os.path.join(base_path, c, '.csv')):
            imagens.append(p)
            labels.append(c)
    
    return np.asarray(imagens), labels

def load_dataset_part(step, thresh):
    train_X = pd.io.parsers.read_csv(os.path.join(base_path_parts, "%.2d_%.3d_train.csv"%(step, thresh*100)))
    test_x = pd.io.parsers.read_csv(os.path.join(base_path_parts, "%.2d_%.3d_test.csv"%(step, thresh*100)))

    train_Y = train_X['Label']
    test_y = test_x['Label']
    
    return train_X, test_x, train_Y, test_y

def load_balance_class_parts(step, thresh):
    train_X, test_x, train_Y, test_y = load_dataset_part(step, thresh)
    train_X = train_X.drop(columns=['Label'])
    test_x = test_x.drop(columns=['Label'])
    
    ##Balanceameto dos dados de treino
    undersample = RandomUnderSampler(sampling_strategy='majority')
    train_under_X, train_under_Y = undersample.fit_resample(train_X, train_Y)
    
    
    lb = LabelBinarizer()
    min_max_scaler = preprocessing.MinMaxScaler()
    train_under_X = min_max_scaler.fit_transform(train_under_X)
    train_under_Y = lb.fit_transform(train_under_Y)
    
    ##Balanceamento dos dados de test
    undersample = RandomUnderSampler(sampling_strategy='majority')
    test_under_X, test_under_Y = undersample.fit_resample(test_x, test_y)
    
    lb = LabelBinarizer()
    min_max_scaler = preprocessing.MinMaxScaler()
    test_under_Y = lb.fit_transform(test_under_Y)
    test_under_X = min_max_scaler.fit_transform(test_under_X)
    
    return train_under_X, train_under_Y, test_under_X, test_under_Y 


def train_ml_algorithm(X_train, y_train, methodName):

    search = None
    # KNN
    # https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html#sklearn.neighbors.KNeighborsClassifier
    if (methodName == 'KNN'): 
        parameters = {
            "n_neighbors" : [1, 3, 5, 10, 15, 20],
            "weights": ['uniform', 'distance'],
            "algorithm": ['ball_tree', 'kd_tree', 'brute'],
            "leaf_size": [5, 15, 25, 35, 45, 55],
            'p': [10, 20, 40],
            "metric": ['euclidean', 'manhattan', 'chebyshev', 'minkowski'],
            # 'metric_params': [None],
            'n_jobs': [20]
        }
        search  = GridSearchCV( 
            estimator=KNeighborsClassifier(),
            param_grid=parameters,
            verbose=1,
            cv=ShuffleSplit(test_size=0.01, n_splits=1, random_state=0),
            # use_gpu=True,
            scoring='accuracy',
            n_jobs=-1
        )
    
    # Decision Tree
    # https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier
    elif (methodName == 'DTree'): 
        parameters = {
            'criterion': ["gini", "entropy"],
            'splitter': ['best','random'],
            # 'max_depth': [None],
            # 'min_samples_split': [None],
            # 'min_samples_leaf': [None],
            # 'min_weight_fraction_leaf': [None],
            'max_features': ['auto', 'sqrt', 'log2'],
            # 'random_state': [None],
            # 'max_leaf_nodes': [None],
            # 'min_impurity_decrease': [None],
            'class_weight': [None, 'balanced'],
            # 'ccp_alpha': [None]
        }
        search  = GridSearchCV( 
            estimator=DecisionTreeClassifier(),
            param_grid=parameters,
            scoring='accuracy',
            # use_gpu=True,
            cv=ShuffleSplit(test_size=0.01, n_splits=1, random_state=0),
            verbose=1,
            n_jobs=-1
        )
        
    
    # SVM Linear
    # https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html#sklearn.svm.LinearSVC
    elif (methodName == "SVMLinear"):  
        parameters = {
            'penalty': ['l1','l2'],
            # 'loss': ['hinge', 'squared_hinge'],
            'dual': [True, False],
            # 'tol': [1],
            'C': np.arange(0.01,100,10),
            'multi_class': ['ovr', 'crammer_singer'],
            'fit_intercept': [True, False],
            # 'intercept_scaling': [None],
            'class_weight': [None, 'balanced'],
            'verbose': [0],
            # 'random_state': [None, 1, 3, 5, 7],
            'max_iter': [1, 10, 50, 100, 200]
        }
        search  = GridSearchCV( 
            estimator=LinearSVC(),
            param_grid=parameters,
            cv=ShuffleSplit(test_size=0.01, n_splits=1, random_state=0),
            scoring='accuracy',
            n_jobs=-1
        )

    # SVM Nu
    # https://scikit-learn.org/stable/modules/generated/sklearn.svm.NuSVC.html#sklearn.svm.NuSVC
    elif (methodName == 'SVMNu'):  
        parameters = {
            'nu': np.arange(0.1,1.1,0.1),
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid'], # 'precomputed' requires an square matrix
            'degree': [1, 2, 3, 4, 5],
            'gamma': ['scale', 'auto'],
            # 'coef0': [None],
            'shrinking': [True, False],
            'probability': [True],
            # 'tol': [None],
            # 'cache_size': [None],
            'class_weight': [None, 'balanced'],
            'verbose': [0],
            'max_iter': [-1],
            'decision_function_shape': ['ovo', 'ovr'],
            'break_ties': [True, False],
            # 'random_state': [None],
        }
        search  = GridSearchCV( 
            estimator=NuSVC(),
            param_grid=parameters,
            cv=ShuffleSplit(test_size=0.01, n_splits=1, random_state=0),
            scoring='accuracy',
            n_jobs=-1
        )

    # SVM C
    # https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC
    elif (methodName == 'SVMC'):  
        parameters = {
            'C': [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1],
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            'degree': [1, 2, 3, 4, 5],
            'gamma': ['scale', 'auto'],
            # 'coef0': [1, 2, 3],
            'shrinking': [True, False],
            'probability': [True, False],
            # 'tol': [None],
            # 'cache_size': [None],
            'class_weight': [None, 'balanced'],
            'verbose': [0],
            'max_iter': [-1],
            'probability':[True],
            'decision_function_shape': ['ovo', 'ovr'],
            # 'break_ties': [True, False],
            # 'random_state': [None],
        }
        search  = GridSearchCV( 
            estimator=SVC(),
            param_grid=parameters,
            cv=ShuffleSplit(test_size=0.01, n_splits=1, random_state=0),
            scoring='accuracy',
            n_jobs=-1
        )
    
    

    # Discriminant Analysis (Linear)
    # https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html#sklearn.discriminant_analysis.LinearDiscriminantAnalysis
    elif (methodName == 'DAnalysisLinear'): 
        parameters = {
            'solver': ['svd', 'lsqr', 'eigen'],
            'shrinkage': ['auto', 0.2, 0.4, 0.6,0.8, 1],
            # 'priors': [None],
            # 'n_components': [None, 10, 20, 40, 50, 100, 200],
            'store_covariance': [True, False],
            # 'tol': [None],
            # 'covariance_estimator': [None],
        }
        search  = GridSearchCV( 
            estimator=LinearDiscriminantAnalysis(),
            param_grid=parameters,
            cv=ShuffleSplit(test_size=0.01, n_splits=1, random_state=0),
            scoring='accuracy',
            n_jobs=-1
        )
    
    # SGD
    # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html?highlight=sgd#sklearn.linear_model.SGDClassifier
    elif (methodName == 'SGD'): 
        parameters = {
            # 'penalty': ['l2', 'l1', 'elasticnet'],
            'alpha': [0.0001, 0.001, 0.01, 0.1, 1],
            'epsilon':[0.01, 0.1, 1]
            # 'l1_ratio': [0.10, 0.15, 0.2, 0.25],
            # 'fit_intercept': [True, False],
            # 'max_iter': [1000, 1200,1400],
            # 'shuffle': [True, False],
            # 'early_stopping': [True],
            # 'n_iter_no_change': [3],
        }
        lr = SGDClassifier(loss='hinge')
        search  = TuneGridSearchCV( 
            estimator=lr,
            param_grid=parameters,
            cv=ShuffleSplit(test_size=0.01, n_splits=1, random_state=0),
            scoring='accuracy',
            n_jobs=-1
        )

    # Discriminant Analysis (Quadratic)
    # https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis.html#sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis
    elif (methodName == 'DAnalysisQuadratic'): 
        parameters = {
            # 'priors': [None],
            'reg_param': [0.1, 0.2, 0.3],
            'store_covariance': [True, False],
            # 'tol': [None],
        }
        search  = GridSearchCV( 
            estimator=QuadraticDiscriminantAnalysis(),
            param_grid=parameters,
            cv=ShuffleSplit(test_size=0.01, n_splits=1, random_state=0),
            scoring='accuracy',
            n_jobs=-1
        )

    # Naive Bayes
    # https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html#sklearn.naive_bayes.GaussianNB
    elif (methodName == 'NBayes'): 
        parameters = {
            # 'priors': None,
            "var_smoothing" : np.logspace(0,-9, num=100),
        }
        search  = GridSearchCV( 
            estimator=GaussianNB(),
            param_grid=parameters,
            cv=ShuffleSplit(test_size=0.01, n_splits=1, random_state=0),
            scoring='accuracy',
            n_jobs=-1
        )


    # AdaBoost
    # https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html?highlight=adaboost#sklearn.ensemble.AdaBoostClassifier
    elif (methodName == 'AdaBoost'): 
        parameters = {
            # 'base_estimator': [None],
            "n_estimators" : [10, 50, 100, 150],
            'learning_rate': [0.5, 1, 1.5],
            'algorithm': ['SAMME', 'SAMME.R'],
            # 'random_state': [3,5,7,9],
        }
        search  = GridSearchCV( 
            estimator=AdaBoostClassifier(),
            param_grid=parameters,
            cv=ShuffleSplit(test_size=0.01, n_splits=1, random_state=0),
            scoring='accuracy',
            n_jobs=-1
        )

    # RandomForest
    # https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html?highlight=random#sklearn.ensemble.RandomForestClassifier
    elif (methodName == 'RandomForest'): 
        parameters = {
            "n_estimators" : [10, 100, 1000],
            'criterion': ["gini", "entropy"],
            'max_depth': [None],
            "max_features" : ['sqrt', 'log2'],
            'verbose': [0],
            'class_weight': ['balanced','balanced_subsample'],
        }
        search  = GridSearchCV( 
            estimator=RandomForestClassifier(),
            param_grid=parameters,
            cv=ShuffleSplit(test_size=0.01, n_splits=1, random_state=0),
            scoring='accuracy',
            n_jobs=-1
        )
    # Bagging
    # https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingClassifier.html#sklearn.ensemble.BaggingClassifier
    elif (methodName == 'Bagging'): 
        parameters = {
            "n_estimators" : [10, 50, 100, 150, 100],
            'verbose': [0],
        }
        search  = GridSearchCV( 
            estimator=BaggingClassifier(),
            param_grid=parameters,
            cv=ShuffleSplit(test_size=0.01, n_splits=1, random_state=0),
            scoring='accuracy',
            n_jobs=-1
        )
    # ExtraTreesClassifier
    # https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html#sklearn.ensemble.ExtraTreesClassifier
    elif (methodName == 'ExtraTrees'): 
        parameters = {
            "n_estimators" : [10, 50, 100, 150, 100],
            'criterion': ["gini", "entropy"],
            "max_features" : ['sqrt', 'log2'],
            'class_weight': ['balanced', 'balanced_subsample'],
            'verbose': [0],
        }
        search  = GridSearchCV( 
            estimator=ExtraTreesClassifier(),
            param_grid=parameters,
            cv=ShuffleSplit(test_size=0.01, n_splits=1, random_state=0),
            scoring='accuracy',
            n_jobs=-1
        )
    # GradientBoostingClassifier
    # https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html#sklearn.ensemble.GradientBoostingClassifier
    elif (methodName == 'GradientBoosting'): 
        parameters = {
            'loss': ['deviance', 'exponential'],
            "n_estimators" : [10, 50, 100, 150, 100],
            'criterion': ["squared_error" ],
            "max_features" : ['sqrt', 'log2'],
            'verbose': [0],
        }
        search  = GridSearchCV( 
            estimator=GradientBoostingClassifier(),
            param_grid=parameters,
            cv=ShuffleSplit(test_size=0.01, n_splits=1, random_state=0),
            scoring='accuracy',
            n_jobs=-1
        )

    # StackingClassifier
    # https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingClassifier.html#sklearn.ensemble.StackingClassifier
    elif (methodName == 'Stacking'): 
        parameters = {
            'stack_method': ['predict_proba', 'decision_function', 'predict'],
            "passthrough" : [True, False],
            'verbose': [0],
        }
        clf1 = KNeighborsClassifier(n_neighbors=1)
        clf2 = RandomForestClassifier(random_state=1)
        clf3 = GaussianNB()
        lr = LogisticRegression()
        search  = GridSearchCV( 
            estimator=StackingClassifier(estimators=[clf1, clf2, clf3],  
                          final_estimator=lr),
            param_grid=parameters,
            cv=ShuffleSplit(test_size=0.01, n_splits=1, random_state=0),
            scoring='accuracy',
            n_jobs=-1
        )

    # VotingClassifier
    # https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingClassifier.html#sklearn.ensemble.VotingClassifier
    elif (methodName == 'Voting'): 
        parameters = {
            'voting': ['hard', 'soft'],
            "flatten_transform" : [True, False],
            'verbose': [0],
        }
        search  = GridSearchCV( 
            estimator=VotingClassifier(),
            param_grid=parameters,
            cv=ShuffleSplit(test_size=0.01, n_splits=1, random_state=0),
            scoring='accuracy',
            n_jobs=-1
        )

    # HistGradientBoostingClassifier
    # https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html#sklearn.ensemble.HistGradientBoostingClassifier
    elif (methodName == 'HistGradient'): 
        parameters = {
            'loss': ['binary_crossentropy', 'categorical_crossentropy'],
        }
        search  = GridSearchCV( 
            estimator=HistGradientBoostingClassifier(),
            param_grid=parameters,
            cv=ShuffleSplit(test_size=0.01, n_splits=1, random_state=0),
            scoring='accuracy',
            n_jobs=-1
        )

    else:
        results = None
    
    
    if (search != None):
        results = search.fit(X_train, y_train.ravel())

    return results
    

def save_parts_proc(part):
    with open(os.path.join(save_metrics_path, "TraditionalML/parts.txt"), mode="a") as f:
        f.write(f"{part}\n")
        
def load_parts_proc():
    parts = []
    with open(os.path.join(save_metrics_path, "TraditionalML/parts.txt"), mode="r") as f:
        parts = f.readlines()

    parts = [p.replace("\n", "") for p in parts]
    
    return parts

if __name__ == '__main__':  
    for thresh in [0.2]:
        for partition in range(38,101):
            X_feat_train, train_under_Y, X_feat_test, test_under_Y = load_balance_class_parts(partition, thresh)
            for methodName in methodsNames:
                print(bcolors.OKGREEN + f"{methodName}: Step {partition}" + bcolors.ENDC)

                print(bcolors.OKCYAN + "Trainning " + methodName + str(thresh*100) + bcolors.ENDC)
                start_train = time.time()
                initClassifier = train_ml_algorithm(X_feat_train, train_under_Y, methodName)
                # calibrator = CalibratedClassifierCV(initClassifier, cv='p   refit')
                # model=calibrator.fit(X_feat_train, train_under_Y)
                y_pred_train  = initClassifier.predict(X_feat_train)
                y_proba_train = initClassifier.predict_proba(X_feat_train)[:, 1]
                # y_pred_train = model.predict_proba(X_feat_train)
                # y_proba_train = model.predict_proba(X_feat_train)[:, 1]
                runtimeTrain = time.time() - start_train
                print(bcolors.OKCYAN + methodName + str(thresh*100) + " Trained in %2.2f seconds"%(runtimeTrain) + bcolors.ENDC)


                print(bcolors.OKCYAN + "Testing " + methodName + str(thresh*100) + bcolors.ENDC)
                start_test = time.time()
                y_pred_test = initClassifier.predict(X_feat_test)
                y_proba_test = initClassifier.predict_proba(X_feat_test)[:, 1]
                runtimeTest = time.time() - start_test
                print(bcolors.OKCYAN + methodName + " Tested in %2.2f seconds"%(runtimeTest) + bcolors.ENDC) 
                

                calculateMeasures(y_pred_train.reshape(y_pred_train.shape[0], 1), train_under_Y, y_proba_train, \
                    y_pred_test.reshape(y_pred_test.shape[0], 1), test_under_Y, y_proba_test, partition, methodName, thresh)

