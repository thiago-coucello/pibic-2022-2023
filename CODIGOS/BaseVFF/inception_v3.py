from operator import index
from this import d
import tensorflow as tf
from keras import backend as K
import numpy as np
from PIL import Image
import pandas as pd
import glob
import time
import datetime
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.utils import to_categorical
#pip install imbalanced-learn
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

from sklearn.metrics import roc_curve, auc, roc_auc_score, confusion_matrix
from tensorflow.keras.models import load_model

tf.config.list_physical_devices('GPU')

import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

##Variaveis globais
save_metrics_path = "../Metrics/"
save_net_name_test = "inception_test.csv"
save_net_name_train = "inception_train.csv"
base_path_parts = "../Partitions/"
files_parts = os.listdir(base_path_parts)
runtimeTrain = 0.0
runtimeTest = 0.0

##Parametros da CNNs
batch_size   = 64
input_shape  = (80, 80, 3)
input_size   = (80, 80)
alpha        = 1e-5
epoch        = 100

lr_reduce = ReduceLROnPlateau(monitor='val_accuracy', factor=0.1, min_delta=alpha, patience=4, verbose=1)
early = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5, mode='max')

def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall_keras = true_positives / (possible_positives + K.epsilon())
    return recall_keras


def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision_keras = true_positives / (predicted_positives + K.epsilon())
    return precision_keras

def specificity(y_true, y_pred):
    tn = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
    fp = K.sum(K.round(K.clip((1 - y_true) * y_pred, 0, 1)))
    return tn / (tn + fp + K.epsilon())

def fmeasure(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2 * ((p * r) / (p + r + K.epsilon()))

# Netavie Predictive Error
def npv(y_true, y_pred):
    tn = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
    fn = K.sum(K.round(K.clip(y_true * (1 - y_pred), 0, 1)))
    return tn / (tn + fn + K.epsilon())

# Matthews Correlation_Coefficient
def mcc(y_true, y_pred):
    tp = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    tn = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
    fp = K.sum(K.round(K.clip((1 - y_true) * y_pred, 0, 1)))
    fn = K.sum(K.round(K.clip(y_true * (1 - y_pred), 0, 1)))

    num = tp * tn - fp * fn
    den = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    return num / K.sqrt(den + K.epsilon())



METRICS = [
    "accuracy",
    tf.keras.metrics.Precision(name='precision'),
    tf.keras.metrics.Recall(name='recall'),
    specificity,
    fmeasure,
    npv,
    mcc,
    tf.keras.metrics.AUC(name='auc',  curve='ROC'),
    tf.keras.metrics.TruePositives(name='tp'),
    tf.keras.metrics.FalsePositives(name='fp'),
    tf.keras.metrics.TrueNegatives(name='tn'),
    tf.keras.metrics.FalseNegatives(name='fn'), 
]

def calculateMeasuresTest(history_net, folder):
    metricsTest = pd.DataFrame()
    idx = np.argmax(history_net.history['val_accuracy'])

    # TRAIN RESULTS
    metricsTest['folder'] = [folder]
    metricsTest['epoch'] = [idx]
    metricsTest['accuracy'] = history_net.history['val_accuracy'][idx]
    metricsTest['precision'] = history_net.history['val_precision'][idx]
    metricsTest['sensitivity'] = history_net.history['val_recall'][idx]
    metricsTest['specificity'] = history_net.history['val_specificity'][idx]
    metricsTest['fmeasure'] = history_net.history['val_fmeasure'][idx]
    metricsTest['npv'] = history_net.history['val_npv'][idx]
    metricsTest['mcc'] = history_net.history['val_mcc'][idx]
    metricsTest['auc'] = history_net.history['val_auc'][idx]
    metricsTest['tn'] = history_net.history['val_tn'][idx]
    metricsTest['fp'] = history_net.history['val_fp'][idx]
    metricsTest['fn'] = history_net.history['val_fn'][idx]
    metricsTest['tp'] = history_net.history['val_tp'][idx]
    metricsTest['runtime'] = [runtimeTest]

    if os.path.exists(os.path.join(save_metrics_path, save_net_name_test)):
        metricsTest.to_csv(os.path.join(save_metrics_path, save_net_name_test), sep=',', mode='a', index=False, header=False)
    else:
        metricsTest.to_csv(os.path.join(save_metrics_path, save_net_name_test), sep=',', index=False)

   
#Funções importantes
def calculateMeasuresTrain(history_net, folder):
    metricsTrain = pd.DataFrame()
    idx = np.argmax(history_net.history['accuracy'])

    # TEST RESULTS
    metricsTrain['folder'] = [folder]
    metricsTrain['epoch'] = [idx]
    metricsTrain['accuracy'] = history_net.history['accuracy'][idx]
    metricsTrain['precision'] = history_net.history['precision'][idx]
    metricsTrain['sensitivity'] = history_net.history['recall'][idx]
    metricsTrain['specificity'] = history_net.history['specificity'][idx]
    metricsTrain['fmeasure'] = history_net.history['fmeasure'][idx]
    metricsTrain['npv'] = history_net.history['npv'][idx]
    metricsTrain['mcc'] = history_net.history['mcc'][idx]
    metricsTrain['auc'] = history_net.history['auc'][idx]
    metricsTrain['tn'] = history_net.history['tn'][idx]
    metricsTrain['fp'] = history_net.history['fp'][idx]
    metricsTrain['fn'] = history_net.history['fn'][idx]
    metricsTrain['tp'] = history_net.history['tp'][idx]
    metricsTrain['runtime'] = [runtimeTrain]

    if os.path.exists(os.path.join(save_metrics_path, save_net_name_train)):
        metricsTrain.to_csv(os.path.join(save_metrics_path, save_net_name_train), sep=',', mode='a', index=False, header=False)
    else:
        metricsTrain.to_csv(os.path.join(save_metrics_path, save_net_name_train), sep=',', index=False)  


def select_image(filename):
    image = Image.open(filename) #load image from file
    image = image.convert('RGB') #convert to RGB, if this option needed
    image = image.resize((80,80)) #resize image to 80x80, according to the dataset
    return np.asarray(image)

def load_dataset(base_path):
    imagens, labels = list(), list()
    classes = os.listdir(base_path)
    for c in classes:
        for p in glob.glob(os.path.join(base_path, c, '*.bmp')):
            imagens.append(p)
            labels.append(c)
    
    return np.asarray(imagens), labels

def load_dataset_part(folder_name):
    
    train_X, train_Y = load_dataset(os.path.join(folder_name, "train"))
    test_x, test_y = load_dataset(os.path.join(folder_name, "test"))

    #images = np.array(images)/255.0
    train_Y, test_y = np.array(train_Y), np.array(test_y) 
    
    #(train_X, test_x, train_Y, test_y) = train_test_split(images, labels, test_size=0.20, stratify=labels)
    
    return train_X, test_x, train_Y, test_y


def laod_balance_class_parts(folder_name):
    train_X, test_x, train_Y, test_y = load_dataset_part(folder_name)
    
    ##Balanceameto dos dados de treino
    undersample = RandomUnderSampler(sampling_strategy='majority')
    train_under_X, train_under_Y = undersample.fit_resample(train_X.reshape(-1,1), train_Y)
    train_under_X = [select_image(p[0]) for p in train_under_X]
    train_under_X = np.array(train_under_X)/255.0
    
    lb = LabelBinarizer()
    train_under_Y = lb.fit_transform(train_under_Y)
    train_under_Y = to_categorical(train_under_Y)
    
    ##Balanceamento dos dados de test
    undersample = RandomUnderSampler(sampling_strategy='majority')
    test_under_X, test_under_Y = undersample.fit_resample(test_x.reshape(-1,1), test_y)
    test_under_X = [select_image(p[0]) for p in test_under_X]
    test_under_X = np.array(test_under_X)/255.0
    
    lb = LabelBinarizer()
    test_under_Y = lb.fit_transform(test_under_Y)
    test_under_Y = to_categorical(test_under_Y)
    
    return train_under_X, train_under_Y, test_under_X, test_under_Y 



def model_inception():
    model_inception_v3 = tf.keras.applications.inception_v3.InceptionV3(include_top=False, weights='imagenet', input_shape=input_shape)
    
    for layer in model_inception_v3.layers[:150]:
        layer.trainable =  False
    
    model_inception_modf = tf.keras.models.Sequential()
    model_inception_modf.add(model_inception_v3)
    model_inception_modf.add(tf.keras.layers.GlobalAveragePooling2D())
    model_inception_modf.add(tf.keras.layers.Dense(128, activation='relu'))
    model_inception_modf.add(tf.keras.layers.BatchNormalization())
    model_inception_modf.add(tf.keras.layers.Flatten())
    model_inception_modf.add(tf.keras.layers.Dropout(0.4))
    model_inception_modf.add(tf.keras.layers.Dense(2, activation='sigmoid'))
    
    model_inception_modf.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001), loss='binary_crossentropy', metrics=METRICS)
    
    
    for layer in model_inception_modf.layers[150:]:
        layer.trainable =  True

    filepath="inception_weights.hdf5"
    checkpoint_inception_v3 = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    
    return model_inception_modf, checkpoint_inception_v3


def predict_test_inception(pred_inception, test_under_Y, folders):
    #y_pred_inception = np.array([np.argmax(p) for p in pred_inception.ravel()])
    y_pred_inception = np.argmax(pred_inception, axis=1)
    y_true = np.argmax(test_under_Y, axis=1)
    y_pred_proba_inception = np.array([np.max(p) for p in pred_inception.ravel()])
    
    #fpr_inception, tpr_inception, _ = roc_curve(test_under_Y.ravel(), y_pred_proba_inception.ravel())
    data_inception = pd.DataFrame()
    
    tn, fp, fn, tp = confusion_matrix(y_true.ravel(), y_pred_inception.ravel(), labels=[0,1]).ravel()

    data_inception = pd.DataFrame(columns=["TN", "FP", "FN", "TP"])
    auc_incp = roc_auc_score(test_under_Y.ravel(), y_pred_proba_inception.ravel())
    
    data_inception["TN"] = [tn]
    data_inception["FP"] = [fp]
    data_inception["FN"] = [fn]
    data_inception["TP"] = [tp]
    data_inception["AUC"] = [auc_incp]
    
    if os.path.exists(os.path.join(save_metrics_path, 'inception/cm_Inception.csv')):
        data_inception.to_csv(os.path.join(save_metrics_path, 'inception/cm_Inception.csv'), sep=',', mode='a', index=False, header=False)
    else:
        data_inception.to_csv(os.path.join(save_metrics_path, 'inception/cm_Inception.csv'), sep=',', index=False)
        
    data_mobile_fpr_tpr = pd.DataFrame(columns=["FPR", "TPR"])
    data_mobile_fpr_tpr["FPR"], data_mobile_fpr_tpr["TPR"], _ = roc_curve(test_under_Y.ravel(), y_pred_proba_inception.ravel())
    
    data_mobile_fpr_tpr.to_csv(os.path.join(save_metrics_path, 'inception/%s_fpr_tpr_Inception.csv'%(str(folders))), sep=',', index=False)


def save_parts_proc(part):
    with open(os.path.join(save_metrics_path, "inception/parts.txt"), mode="a") as f:
        f.write(part+"\n")
        
def load_parts_proc():
    parts = []
    with open(os.path.join(save_metrics_path, "inception/parts.txt"), mode="r") as f:
        parts = f.readlines()

    parts = [p.replace("\n", "") for p in parts]
    
    return parts

if __name__ == '__main__':
    
    log_dir = save_metrics_path + "/inception/" + "log_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    parts = load_parts_proc()
    
    for folder in files_parts:  
        if folder in parts:
            continue

        csv_logger = tf.keras.callbacks.CSVLogger(os.path.join(save_metrics_path, "inception/%0.2d-epoch-results.csv"%(int(folder))), separator=',', append=True)

        print(f"Step: {folder}")
       
        
        initModel, checkpoint = model_inception()
        train_under_X, train_under_Y, test_under_X, test_under_Y = laod_balance_class_parts(os.path.join(base_path_parts, folder))
        print("Trainning InceptionNet")
        start_train = time.time()
        with tf.device('/device:GPU:0'):
            history_net = initModel.fit(train_under_X,
                                    train_under_Y,
                                    steps_per_epoch=(len(train_under_X) // batch_size),
                                    validation_steps = len(test_under_X) // batch_size,
                                    batch_size = batch_size,
                                    epochs=epoch, 
                                    validation_data=(test_under_X, test_under_Y), 
                                    callbacks=[early, checkpoint, lr_reduce,csv_logger])
        
        runtimeTrain = time.time() - start_train
        print("EfficientNet Trained in %2.2f seconds"%(runtimeTrain))

        dependencies = {
            'precision': precision,
            'recall': recall,
            'fmeasure': fmeasure,
            'specificity': specificity,
            'npv': npv,
            'mcc': mcc,
        }

        print("Testing InceptionNet")
        model = load_model('inception_weights.hdf5', custom_objects=dependencies)
        start_test = time.time()
        pred_inception = model.predict(test_under_X)
        runtimeTest = time.time() - start_test
        print("InceptionNet Tested in %2.2f seconds"%(runtimeTest)) 


        predict_test_inception(pred_inception, test_under_Y, folder)
        

        calculateMeasuresTrain(history_net, folder)
        calculateMeasuresTest(history_net, folder)
        save_parts_proc(folder)
