import warnings
warnings.filterwarnings('ignore')
import os
import sys
sys.path.append('..')
import sklearn
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.feature_selection import RFE, RFECV
from sklearn.linear_model import Lasso, LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, balanced_accuracy_score, recall_score, f1_score, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import shap
# from utils import load_VBM, load_ASL
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import pickle
import keras
import tensorflow as tf
from keras.layers import Dense, Input
from keras.models import Model
from keras.layers.core import Dropout
from keras.layers import LeakyReLU
from keras.layers import BatchNormalization
from keras import initializers

def DNN_model(input_dim, numClass):
    print('****DNN Model****')
    inputs = Input(shape = (input_dim,), name = 'inputs_layer')
    x = Dense(256, kernel_initializer = initializers.glorot_normal(), name = 'hidden_layer1')(inputs)
    x = BatchNormalization(name = 'bn1')(x)
    x = LeakyReLU()(x)
    x = Dropout(0.5, name = 'dropout1')(x)
    x = Dense(4, kernel_initializer = initializers.glorot_normal(), name = 'hidden_layer2')(x)
    x = BatchNormalization(name = 'bn2')(x)
    x = LeakyReLU()(x)
    x = Dropout(0.5, name = 'dropout2')(x)

    predictions = Dense(units = numClass, activation = 'sigmoid', name = 'prediction_layer')(x)

    model = Model(inputs = inputs, outputs = predictions)
    return model

def eval_model(y_predprob, Y_test):
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, recall_score, f1_score, confusion_matrix, roc_auc_score
    # y_pred = np.argmax(y_predprob, axis=1)
    y_pred = np.where(y_predprob>0.5, 1, 0)
    result = {}
    result['acc'] = accuracy_score(Y_test, y_pred)
    result['bal_acc'] = balanced_accuracy_score(Y_test, y_pred)

    tn, fp, fn, tp = confusion_matrix(Y_test, y_pred).ravel()
    result['sen'] = tp / (tp + fn)
    # tp / (tp + fn) = recall_score(Y_test, y_pred, average='macro')
    result['spe'] = tn / (fp + tn)

    result['auc'] = roc_auc_score(Y_test, y_pred)
    result['f1'] = f1_score(Y_test, y_pred)
    result['weighted-f1'] = f1_score(Y_test, y_pred, average='weighted')
    result['tn'] = tn
    result['fp'] = fp
    result['fn'] = fn
    result['tp'] = tp
    return result



res = {}
res_soft = {}
asl = np.load('../data/asl.npy')
vbm = np.load('../data/vbm.npy', allow_pickle=True)
data = np.concatenate((vbm, asl), 1)
labels = np.load('../data/labels.npy')
seed = 2023
outputdir = './vbm+asl/'
skf1 = StratifiedKFold(n_splits = 10, shuffle=True, random_state = seed)
for n_fold, (train_val, test) in enumerate(skf1.split(labels, labels)):
    print('-'*7, n_fold, '-'*7)
    
    train_val_dataset, test_dataset = data[train_val.tolist()], data[test.tolist()]
    train_val_labels = labels[train_val]
    train_val_index = np.arange(len(train_val_dataset))

    train, val, _, _ = train_test_split(train_val_index, train_val_labels, test_size=0.11, shuffle=True, stratify=train_val_labels,random_state=seed)
    train_dataset, val_dataset = train_val_dataset[train.tolist()], train_val_dataset[val.tolist()]
    # 
    from imblearn.under_sampling import RandomUnderSampler
    rus = RandomUnderSampler(random_state=0,replacement=True)
    X_train_, Y_train = rus.fit_resample(train_dataset, train_val_labels[train])

    # 
    scaler = StandardScaler()
    scaler.fit(X_train_)
    X_train = scaler.transform(X_train_)
    X_val = scaler.transform(val_dataset)
    X_test = scaler.transform(test_dataset)
    Y_val, Y_test = train_val_labels[val], labels[test]
    # FCNN
    bestModelSavePath = outputdir + '/fcnn/best_model_%02i.hdf5' % (n_fold)
    # 
    model = DNN_model(X_train.shape[1], 1)
    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=['accuracy'])
    model.load_weights(bestModelSavePath)
    fcnn_y_predprob = model.predict(X_test) # (26,1)
    fcnn_train = model.predict(X_train)
    # svm_linear
    file = open(outputdir + "/svm_linear/best_model_%02i.pickle" % (n_fold), "rb")
    sl_model = pickle.load(file)
    file.close()
    sl_y_predprob = sl_model.predict_proba(X_test)[:, 1] # (26,2)
    sl_y_predprob = sl_y_predprob.reshape(fcnn_y_predprob.shape[0], 1)
    sl_train = sl_model.predict_proba(X_train)[:, 1]
    sl_train = sl_train.reshape(fcnn_train.shape[0], 1)
    # svm_rbf
    file = open(outputdir + "/svm_rbf/best_model_%02i.pickle" % (n_fold), "rb")
    sr_model = pickle.load(file)
    file.close()
    sr_y_predprob = sr_model.predict_proba(X_test)[:,1] # (26,2)
    sr_y_predprob = sr_y_predprob.reshape(fcnn_y_predprob.shape[0], 1)
    sr_train = sr_model.predict_proba(X_train)[:, 1]
    sr_train = sr_train.reshape(fcnn_train.shape[0], 1)
    # rf
    file = open(outputdir + "/rf/best_model_%02i.pickle" % (n_fold), "rb")
    rf_model = pickle.load(file)
    file.close()
    rf_y_predprob = rf_model.predict_proba(X_test)[:,1] # (26,2)
    rf_y_predprob = rf_y_predprob.reshape(fcnn_y_predprob.shape[0], 1)
    rf_train = rf_model.predict_proba(X_train)[:, 1]
    rf_train = rf_train.reshape(fcnn_train.shape[0], 1)
    predicted_probas = np.concatenate((fcnn_y_predprob, sl_y_predprob, sr_y_predprob, rf_y_predprob), 1)
    
    sv_predicted_proba = np.mean(predicted_probas, axis=1)
    sv_predicted_proba = sv_predicted_proba.reshape(sv_predicted_proba.shape[0], 1) # (26,1)
    yhat = np.zeros((X_test.shape[0], 2))
    yhat[:, 1] = sv_predicted_proba[:, 0]
    yhat[:, 0] = 1 - sv_predicted_proba[:,0] # (26,2)
    res_temp = eval_model(yhat[:,1], Y_test)
    for key in res_temp.keys():
        res_soft.setdefault(key, []).append(res_temp[key])

    
df = pd.DataFrame(data = res_soft)
# df = pd.DataFrame(data = res)
result = df.to_numpy().astype('float')
mean = np.mean(result, 0)
std = np.std(result, 0)
a = []
for i in range(11):
    temp = '%.4f±%.4f'%(mean[i], std[i])
    a.append(temp)
df.loc['mean±std'] = a
# df.loc['mean'] = mean
# df.loc['std'] = std
df.to_csv('ttt.csv')

