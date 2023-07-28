from operator import index
from this import d
import tensorflow as tf
from keras import backend as K
import numpy as np
from PIL import Image
import pandas as pd
import csv
import glob
import time
import datetime
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import math
from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.utils import to_categorical
#pip install imbalanced-learn
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

from sklearn.metrics import roc_curve, auc, roc_auc_score, confusion_matrix
from tensorflow.keras.models import load_model


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

tf.config.list_physical_devices('GPU')

import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['CUDA_VISIBLE_DEVICES'] = "1"

##Variaveis globais
#save_metrics_path = "../Datasets/DatasetBalanced2/Results/csvs/"
#save_csvs_path = "../Datasets/DatasetBalanced2/Results/csvs/"
#save_nets_path = "../Datasets/DatasetBalanced2/Results/nets/"
#base_path_parts = "../Datasets/DatasetBalanced2/Partitions/"

# 05 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95 100
subsets = ["05", "10", "15", "20", "25", "30", "35", "40", "45", "50", "55", "60", "65", "70", "75", "80", "85", "90", "95", "100"]

runtimeTrain = 0.0
runtimeTest = 0.0

# , 'VGG16', 'VGG19', 'ResNet50', 'ResNet50V2', 'ResNet101', 'ResNet101V2', 'ResNet152', 'ResNet152V2', \
#     , 'InceptionResNetV2', , 'MobileNetV2', 'DenseNet121','DenseNet169', , 'EfficientNetB0', 'EfficientNetB1', 

#    ,  \
# 'EfficientNetV2B0', 'EfficientNetV2B1', 'EfficientNetV2B2', 'EfficientNetV2B3', 'EfficientNetV2S', 'EfficientNetV2M', 'EfficientNetV2L', \
# 'EfficientNetV2B0', 'EfficientNetB2', 'EfficientNetB3', , 'EfficientNetB5', 'EfficientNetB6', 'EfficientNetB7'



# Refinar:  

# 'InceptionV3':    Dense 512  Dropout  0.1 Freeze 0.3
# 'Xception':       Dense 1024   Dropout 0.1  Freeze 0.3
# 'EfficientNetB4': Dense 1024   Dropout 0.1  Freeze 0.3
# 'Resnet101V2':    Dense  128   Dropout 0.1  Freeze 0.5
# 'ResNet50V2':     Dense 128    Dropout 0.1  Freeze 0.5
# 'MobileNet':      Dense 1024   Dropout 0.1  Freeze 0.5
# 'ResNet152V2':    Dense 128    Dropout 0.1  Freeze 0.3
# 'DenseNet201':    Dense 128    Dropout 0.1  Freeze 0.3
# 'MobileNetV2':    Dense 128    Dropout 0.1  Freeze 0.3
methodsNames = ['MobileNetV2', 'VGG16', 'VGG19', 'ResNet50', "InceptionV3"] 
# 'VGG16', 'VGG19', 'ResNet50', 'ResNet50V2', 'ResNet101', 'ResNet101V2', 'ResNet152', 'ResNet152V2', 'DenseNet201', 'Xception', 'EfficientNetB4'

# DONE: 'DenseNet201', 
# 'MobileNet' (1-50), 'ResNet101V2' (1-50), "ResNet50V2" (1-50), "ResNet152V2" (1-50), 
# "MobileNetV2" (1-40), "VGG16" (1-40), "VGG19" (1-40), "ResNet50" (1-40),
# "ResNet101" (1-50), "ResNet152" (1-50), "Xception" (1-50), "EfficientNetB4" (1-50), 
# "InceptionV3" (1-40)

##Parametros da CNNs
batch_size   = 32
input_shape  = (128, 128, 3)
alpha        = 1e-5
epoch        = 60

lr_reduce = ReduceLROnPlateau(monitor='val_accuracy', factor=0.1, min_delta=alpha, patience=3, verbose=0)
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

def f1_score(y_true, y_pred):
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
    f1_score,
    npv,
    mcc,
    tf.keras.metrics.AUC(name='auc',  curve='ROC'),
    tf.keras.metrics.TruePositives(name='tp'),
    tf.keras.metrics.FalsePositives(name='fp'),
    tf.keras.metrics.TrueNegatives(name='tn'),
    tf.keras.metrics.FalseNegatives(name='fn'), 
]


   
#Funções importantes
def calculateMeasures(history_net, folder, methodName, denseNum, dropOut, freezePercentage, batchsize):
    metrics = pd.DataFrame()
    idx = np.argmax(history_net.history['val_accuracy'])

    metrics['folder'] = [folder]
    metrics['network'] = [methodName]
    metrics['DenseNum'] = [denseNum]
    metrics['DropOut'] = [dropOut]
    metrics['FreezePercentage'] = [freezePercentage]
    metrics['BatchSize'] = [batchsize]

    # TEST RESULTS
    
    metrics['epoch'] = [idx]
    metrics['accuracy'] = history_net.history['accuracy'][idx]
    metrics['precision'] = history_net.history['precision'][idx]
    metrics['sensitivity'] = history_net.history['recall'][idx]
    metrics['specificity'] = history_net.history['specificity'][idx]
    metrics['f1_score'] = history_net.history['f1_score'][idx]
    metrics['npv'] = history_net.history['npv'][idx]
    metrics['mcc'] = history_net.history['mcc'][idx]
    metrics['auc'] = history_net.history['auc'][idx]
    metrics['tn'] = history_net.history['tn'][idx]
    metrics['fp'] = history_net.history['fp'][idx]
    metrics['fn'] = history_net.history['fn'][idx]
    metrics['tp'] = history_net.history['tp'][idx]
    metrics['runtime'] = [runtimeTrain]
    # TRAIN RESULTS

    metrics['val_accuracy'] = history_net.history['val_accuracy'][idx]
    metrics['val_precision'] = history_net.history['val_precision'][idx]
    metrics['val_sensitivity'] = history_net.history['val_recall'][idx]
    metrics['val_specificity'] = history_net.history['val_specificity'][idx]
    metrics['val_f1_score'] = history_net.history['val_f1_score'][idx]
    metrics['val_npv'] = history_net.history['val_npv'][idx]
    metrics['val_mcc'] = history_net.history['val_mcc'][idx]
    metrics['val_auc'] = history_net.history['val_auc'][idx]
    metrics['val_tn'] = history_net.history['val_tn'][idx]
    metrics['val_fp'] = history_net.history['val_fp'][idx]
    metrics['val_fn'] = history_net.history['val_fn'][idx]
    metrics['val_tp'] = history_net.history['val_tp'][idx]
    metrics['val_runtime'] = [runtimeTest]

    print(bcolors.FAIL + 'ACC: %.2f' %(100*metrics['val_accuracy'][0]) + ' AUC: %.2f' %(100*metrics['val_auc'][0]) + bcolors.ENDC)

    if os.path.exists(os.path.join(save_csvs_path, methodName + '_refined.csv')):
        metrics.to_csv(os.path.join(save_csvs_path, methodName + '_refined.csv'), sep=',', mode='a', index=False, header=False)
    else:
        metrics.to_csv(os.path.join(save_csvs_path, methodName + '_refined.csv'), sep=',', index=False)  


def select_image(filename):
    image = Image.open(filename) #load image from file
    image = image.convert('RGB') #convert to RGB, if this option needed
    image = np.array(image)
    image = tf.image.resize_with_crop_or_pad(image, 80,80) # DEixa imagem quadrada em 80x80
    image = tf.image.resize(image, [input_shape[0], input_shape[1]]) #resize image to 120x120
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

def load_dataset(partition):
    TrainImages, TestImages, TrainLabels, TestLabels = list(), list(), list(), list()

    fName = os.path.join(base_path_parts, partition + '.csv')

    # Read Partition Details
    with open(fName, 'r') as file:
        csvreader = csv.reader(file)
        header = next(csvreader)
        #         IDX         0        1        2        3
        # print(header) # ['Image', 'Class', 'Train', 'Test']
        for row in csvreader:
            # image = np.asarray(np.array(select_image(row[0]))/255.0)
            # print(len(image))
            # Train sample
            if (row[2] == '1'):
                TrainImages.append(row[0])
                TrainLabels.append(row[1] == '1')
            # Test Sample
            elif (row[3] == '1'):
                TestImages.append(row[0])
                TestLabels.append(row[1] == '1')

    TrainImages = np.asarray(TrainImages)
    TestImages = np.asarray(TestImages)

    #Balanceameto dos dados de treino
    undersample = RandomUnderSampler(sampling_strategy='majority')
    TrainImages, TrainLabels = undersample.fit_resample(TrainImages.reshape(-1,1), TrainLabels)
    TrainImages = [select_image(p[0]) for p in TrainImages]
    TrainImages = np.array(TrainImages)/255.0
    # print(TrainImages)

    #Balanceamento dos dados de test
    undersample = RandomUnderSampler(sampling_strategy='majority')
    TestImages, TestLabels = undersample.fit_resample(TestImages.reshape(-1,1), TestLabels)
    TestImages = [select_image(p[0]) for p in TestImages]
    TestImages = np.array(TestImages)/255.0
    # print(TestImages)
    
    lb = LabelBinarizer()
    TrainLabels = to_categorical( lb.fit_transform( np.array(TrainLabels) ), num_classes=2)
    TestLabels  = to_categorical( lb.fit_transform( np.array(TestLabels ) ), num_classes=2)
    
    return TrainImages, TrainLabels, TestImages , TestLabels

def makemodel(folder, methodName, denseNum, dropOut, freezePercentage):
    # create the base pre-trained model
    base_model = eval("tf.keras.applications." + methodName + "(weights='imagenet', include_top=False, input_shape=input_shape)")
    
    # Congela 50% do total de layers da rede
    numLayersFreeze = math.floor(len(base_model.layers)*freezePercentage)
    for layer in base_model.layers[:numLayersFreeze]:
        layer.trainable =  False
    
    model = tf.keras.models.Sequential()
    model.add(base_model)
    model.add(tf.keras.layers.GlobalAveragePooling2D())
    model.add(tf.keras.layers.Dense(denseNum, activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dropout(dropOut))
    model.add(tf.keras.layers.Dense(2, activation='sigmoid'))
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9), loss='binary_crossentropy', metrics=METRICS) # 
    
    for layer in model.layers[numLayersFreeze:]:
        layer.trainable =  True

    # folder + '_' +
    fname = methodName + '_weights.hdf5'
    filepath= os.path.join(save_nets_path, fname)
    checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=0, save_best_only=True, mode='max')
    return model, checkpoint




def predict_test(pred_inception, test_under_Y, folders, methodName):
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
    
    if os.path.exists(os.path.join(save_csvs_path, methodName, '_cm.csv')):
        data_inception.to_csv(os.path.join(save_csvs_path, methodName,  '_cm.csv'), sep=',', mode='a', index=False, header=False)
    else:
        data_inception.to_csv(os.path.join(save_csvs_path, methodName, '_cm.csv'), sep=',', index=False)
        
    data_mobile_fpr_tpr = pd.DataFrame(columns=["FPR", "TPR"])
    data_mobile_fpr_tpr["FPR"], data_mobile_fpr_tpr["TPR"], _ = roc_curve(test_under_Y.ravel(), y_pred_proba_inception.ravel())
    
    data_mobile_fpr_tpr.to_csv(os.path.join(save_csvs_path, methodName, '%s_fpr_tpr.csv'%(str(folders))), sep=',', index=False)



if __name__ == '__main__':
    #  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19
    # 05 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95 100
    for subset_idx in range(0, 20):
        subset = subsets[subset_idx]
  
        save_metrics_path = f"C:\\PIBIC\\2022-2023\\Results\\{subset}\\DL\\csvs\\"
        save_csvs_path = f"C:\\PIBIC\\2022-2023\\Results\\{subset}\\DL\\csvs\\"
        save_nets_path = f"D:\\PIBIC\\2022-2023\\Results\\{subset}\\DL\\nets\\"
        # base_path_parts = "C:\\PIBIC\\2022-2023\\Datasets\\05\\Partitions\\"
        base_path_parts = f"C:\\PIBIC\\2022-2023\\Datasets\\{subset}\\Partitions"

        files_parts = os.listdir(base_path_parts)

        for method in range(0, len(methodsNames)):
            methodName = methodsNames[method]

            if not os.path.exists(os.path.join(save_csvs_path, methodName)):
                os.makedirs(os.path.join(save_csvs_path, methodName))

            # Min: 1 - Max: 101 (Partições de 1 a 100)
            for partition in range(41, 51): # Partições de 41 a 50
                partition = str(partition) 
                for denseNum in [128]: # range(128,128, 128):
                    for dropOut in [0.3]: #0.2,0.3,0.4,0.5
                        for freezePercentage in [0.3]: # 
                            print(bcolors.OKGREEN + f"{methodName}: Subset {subset} Partition {partition} DenseNum {denseNum}, dropout {dropOut}, freezePercentage {freezePercentage} " + bcolors.ENDC)

                            log_dir = save_csvs_path + "/" + methodName + "/" + "log_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                            tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

                            csv_logger = tf.keras.callbacks.CSVLogger(os.path.join(save_csvs_path, methodName, "%0.2d-epoch-results.csv"%(int(partition))), separator=',', append=True)
                        
                            
                            initModel, checkpoint = makemodel(partition, methodName, denseNum, dropOut, freezePercentage)
                            
                            # train_under_X, train_under_Y, test_under_X, test_under_Y = laod_balance_class_parts(os.path.join(base_path_parts, partition))
                            train_under_X, train_under_Y, test_under_X, test_under_Y = load_dataset(partition)
                            
                            print(bcolors.OKCYAN + "Trainning " + methodName + bcolors.ENDC)
                            start_train = time.time()
                            with tf.device('/device:GPU:0'):
                                history_net = initModel.fit(train_under_X,
                                                        train_under_Y,
                                                        steps_per_epoch=(len(train_under_X) // batch_size),
                                                        validation_steps = (len(test_under_X) // batch_size),
                                                        batch_size = batch_size,
                                                        epochs=epoch, 
                                                        validation_data=(test_under_X, test_under_Y), 
                                                        callbacks=[early, checkpoint, lr_reduce,csv_logger])
                            
                            runtimeTrain = time.time() - start_train
                            print(bcolors.OKCYAN + methodName + " Trained in %2.2f seconds"%(runtimeTrain) + bcolors.ENDC)
                            

                            dependencies = {
                                'precision': precision,
                                'recall': recall,
                                'f1_score': f1_score,
                                'specificity': specificity,
                                'npv': npv,
                                'mcc': mcc
                            }

                            print(bcolors.OKCYAN + "Testing " + methodName + bcolors.ENDC)
                            # 

                            if (not os.path.isdir(save_nets_path)):
                                os.mkdir(save_nets_path)

                            fname = methodName + '_weights.hdf5'
                            filepath= os.path.join(save_nets_path, fname)
                            model = load_model(filepath, custom_objects=dependencies)
                            start_test = time.time()
                            pred_inception = model.predict(test_under_X)
                            runtimeTest = time.time() - start_test
                            print(bcolors.OKCYAN + methodName + " Tested in %2.2f seconds"%(runtimeTest) + bcolors.ENDC)
                        
                            calculateMeasures(history_net, partition, methodName, denseNum, dropOut, freezePercentage, batch_size)
                            # calculateMeasuresTest(history_net, step, methodName)
