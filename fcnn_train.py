# -*- coding: utf-8 -*-
import numpy as np
import os
import sys
sys.path.append('..')
import scipy.io as sio
import keras
import pandas as pd
import tensorflow as tf
from keras.layers import Dense, Input
from keras.models import Model
from keras.layers.core import Dropout
from keras.layers import LeakyReLU
from keras.layers import BatchNormalization
from keras import initializers
from keras import backend as K
from sklearn.metrics import roc_auc_score, f1_score, balanced_accuracy_score, confusion_matrix, accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

tf.reset_default_graph()
tf.set_random_seed(2023)

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


# preprocessing data
labels = np.load('../data/labels.npy')
cbf = np.load('../data/asl.npy')
vbm = np.load('../data/vbm.npy', allow_pickle=True)
data = np.concatenate((vbm, cbf),1)
# data = cbf
lr = 0.001
epochs = 300
batch_size = 8
outputdir = './vbm+asl/fcnn/'
if not os.path.exists(outputdir):
    os.makedirs(outputdir)
seed = 2023
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
eval_metrics = np.zeros((skf.n_splits, 11))

for n_fold, (train_val, test) in enumerate(skf.split(labels, labels)):
    print('-'*7, n_fold, '-'*7)
    
    train_val_dataset, test_dataset = data[train_val.tolist()], data[test.tolist()]
    train_val_labels = labels[train_val]
    train_val_index = np.arange(len(train_val_dataset))

    train, val, _, _ = train_test_split(train_val_index, train_val_labels, test_size=0.11, shuffle=True, stratify=train_val_labels, random_state=seed)
    train_dataset, val_dataset = train_val_dataset[train.tolist()], train_val_dataset[val.tolist()]
    # down-sampling
    from imblearn.under_sampling import RandomUnderSampler
    rus = RandomUnderSampler(random_state=0,replacement=True)
    X_train_, Y_train = rus.fit_resample(train_dataset, train_val_labels[train])

    # standardization
    scaler = StandardScaler()
    scaler.fit(X_train_)
    X_train = scaler.transform(X_train_)
    X_val = scaler.transform(val_dataset)
    X_test = scaler.transform(test_dataset)
    Y_val, Y_test = train_val_labels[val], labels[test]

    bestModelSavePath = outputdir+'best_model_%02i.hdf5' % (n_fold)
    model = DNN_model(X_train.shape[1], 1)
    optimizer = keras.optimizers.Adam(learning_rate=lr)
    checkpoint = [
        ModelCheckpoint(bestModelSavePath, monitor='accuracy', verbose=1, save_best_only=True, mode='auto'),
        EarlyStopping(monitor='val_loss', min_delta=0, patience=20),
        ReduceLROnPlateau(monitor='val_acc', patience=5, mode='auto')
    ]
    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=['accuracy'])
    model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_val,Y_val), callbacks=checkpoint)

    y_predprob = model.predict(X_test)
    y_pred = np.where(y_predprob>0.5,1,0)

    tn, fp, fn, tp = confusion_matrix(Y_test, y_pred).ravel()
    eval_metrics[n_fold, 0] = accuracy_score(Y_test, y_pred) # best_test_acc
    eval_metrics[n_fold, 1] = balanced_accuracy_score(Y_test, y_pred) # best_test_bal_acc
    eval_metrics[n_fold, 2] = tp / (tp + fn + 0.000001) # best_test_sen
    eval_metrics[n_fold, 3] = tn / (fp + tn + 0.000001) # best_test_spe
    eval_metrics[n_fold, 4] = roc_auc_score(Y_test, y_pred) # best_test_auc
    eval_metrics[n_fold, 5] = f1_score(Y_test, y_pred) # best_test_f1
    eval_metrics[n_fold, 6] = f1_score(Y_test, y_pred, average='weighted') # best_test_f1
    eval_metrics[n_fold, 7] = tn # best_test_tn
    eval_metrics[n_fold, 8] = fp # best_test_fp
    eval_metrics[n_fold, 9] = fn # best_test_fn
    eval_metrics[n_fold, 10] = tp # best_test_tp
    del model
    K.clear_session()

    """
    print('history:')
    print(history.history)
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)
    plt.plot(epochs, acc, 'bo', label='Trainning acc')
    plt.plot(epochs, val_acc, 'b', label='Vaildation acc')
    plt.legend()

    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Trainning loss')
    plt.plot(epochs, val_loss, 'b', label='Vaildation loss')
    plt.legend()
    plt.show() 
    """
eval_df = pd.DataFrame(eval_metrics)
eval_df.columns = ['ACC','bal_ACC', 'SEN', 'SPE', 'AUC', 'F1','weighted-F1', 'TN', 'FP', 'FN', 'TP']
eval_df.index = ['Fold_%02i' % (i + 1) for i in range(skf.n_splits)]
mean = np.nanmean(eval_metrics, 0)
std = np.nanstd(eval_metrics, 0)
a = []
for i in range(11):
    temp = '%.4f±%.4f'%(mean[i], std[i])
    a.append(temp)
eval_df.loc['mean±std'] = a
print(eval_df)

eval_df.to_csv(outputdir + '/exp_result.csv')
