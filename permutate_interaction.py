import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
import os
import numpy as np
import pandas as pd
import copy
import time
from collections import defaultdict
import itertools 
from scipy import stats
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, balanced_accuracy_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from imblearn.under_sampling import RandomUnderSampler
import pickle
from progress.bar import IncrementalBar
import keras
from keras.layers import Dense, Input
from keras.models import Model
from keras.layers.core import Dropout
from keras.layers import LeakyReLU
from keras.layers import BatchNormalization
from keras import initializers
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" 

def DNN_model(input_dim, numClass):
    # print('****DNN Model****')
    inputs = Input(shape = (input_dim,), name = 'inputs_layer')
    x = Dense(256, kernel_initializer = initializers.glorot_normal(), name = 'hidden_layer1')(inputs)
    x = BatchNormalization(name = 'bn1')(x)
    x = LeakyReLU()(x)
    x = Dropout(0.5, name = 'dropout1')(x)
    x = Dense(4, kernel_initializer = initializers.glorot_normal(), name = 'hidden_layer2')(x)
    x = BatchNormalization(name = 'bn2')(x)
    x = LeakyReLU()(x)
    x = Dropout(0.5, name = 'dropout2')(x)
    x = Dense(4, kernel_initializer = initializers.glorot_normal(), name = 'hidden_layer3')(x)
    x = BatchNormalization(name = 'bn3')(x)
    x = LeakyReLU()(x)
    x = Dropout(0.5, name = 'dropout3')(x)

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

    result['auc'] = roc_auc_score(Y_test, y_predprob)
    result['f1'] = f1_score(Y_test, y_pred)

    result['tn'] = tn
    result['fp'] = fp
    result['fn'] = fn
    result['tp'] = tp
    return result

def interpreting(X_test):
    outputdir = './vbm+asl/'
    n_fold = 4
    # FCNN
    bestModelSavePath = outputdir + '/fcnn/best_model_%02i.hdf5' % (n_fold)
    # 
    model = DNN_model(696, 1)
    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=['accuracy'])
    model.load_weights(bestModelSavePath)
    fcnn_y_predprob = model.predict(X_test) # (26,1)
    # fcnn_train = model.predict(X_train)
    # svm_linear
    file = open(outputdir + "/svm_linear/best_model_%02i.pickle" % (n_fold), "rb")
    sl_model = pickle.load(file)
    file.close()
    sl_y_predprob = sl_model.predict_proba(X_test)[:, 1] # (26,2)
    sl_y_predprob = sl_y_predprob.reshape(fcnn_y_predprob.shape[0], 1)
    # sl_train = sl_model.predict_proba(X_train)[:, 1]
    # sl_train = sl_train.reshape(fcnn_train.shape[0], 1)
    # svm_rbf
    file = open(outputdir + "/svm_rbf/best_model_%02i.pickle" % (n_fold), "rb")
    sr_model = pickle.load(file)
    file.close()
    sr_y_predprob = sr_model.predict_proba(X_test)[:,1] # (26,2)
    sr_y_predprob = sr_y_predprob.reshape(fcnn_y_predprob.shape[0], 1)
    # sr_train = sr_model.predict_proba(X_train)[:, 1]
    # sr_train = sr_train.reshape(fcnn_train.shape[0], 1)
    # rf
    file = open(outputdir + "/rf/best_model_%02i.pickle" % (n_fold), "rb")
    rf_model = pickle.load(file)
    file.close()
    rf_y_predprob = sl_model.predict_proba(X_test)[:,1] # (26,2)
    rf_y_predprob = rf_y_predprob.reshape(fcnn_y_predprob.shape[0], 1)
    # rf_train = rf_model.predict_proba(X_train)[:, 1]
    # rf_train = rf_train.reshape(fcnn_train.shape[0], 1)
    predicted_probas = np.concatenate((fcnn_y_predprob, sl_y_predprob, sr_y_predprob, rf_y_predprob), 1)
    sv_predicted_proba = np.mean(predicted_probas, axis=1)
    sv_predicted_proba = sv_predicted_proba.reshape(sv_predicted_proba.shape[0], 1) # (26,1)
    yhat = np.zeros((X_test.shape[0], 2))
    yhat[:, 1] = sv_predicted_proba[:, 0]
    yhat[:, 0] = 1 - sv_predicted_proba[:,0] # (26,2)
    from keras import backend as K
    K.clear_session()
    return yhat

def fimp(dataset, target, seed, M, key_fea):
    # M = 'acc' or 'auc'
    # 
    yhat = interpreting(dataset)
    res = eval_model(yhat[:, 1], target)
    dataset_performance = res[M]
    np.save('./dataset_performance_%d.npy'%(seed), dataset_performance)
    # 
    df = pd.DataFrame(dataset)
    new_df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    new_dataset = new_df.to_numpy()
    new_yhat = interpreting(new_dataset)
    new_res = eval_model(new_yhat[:, 1], target)
    shuffled_dataset_performance = new_res[M]
    # 
    bar = IncrementalBar('Countdown', max = len(df.columns))
    
    if os.path.exists('./single_feature_performance_%d.npy'%(seed)):
        single_feature_performance = np.load('./single_feature_performance_%d.npy'%(seed), allow_pickle=True).item()
    else:
        single_feature_performance = {}
        import time
        start = time.time()
        for i,feat in enumerate(df.columns):
            bar.next()
            new_fea = copy.deepcopy(df)
            # new_fea[feat] = df[feat].sample(frac=1, random_state=seed).reset_index(drop=True)
            new_fea[feat] = new_df[feat]  # 
            new_fea = new_fea.to_numpy()
            new_fea_yhat = interpreting(new_fea)
            new_fea_res = eval_model(new_fea_yhat[:, 1], target)
            single_feature_performance[feat] = new_fea_res[M]
            bar.finish()
            if feat%100 == 0:
                end = time.time()
                print("time:", end-start)
                start = end
            
        np.save('single_feature_performance_%d.npy'%(seed), single_feature_performance)
    # 
    from collections import defaultdict
    import itertools
    results = defaultdict(dict)
    for feature, other_feature in itertools.product(df.columns, df.columns):
        if feature not in key_fea:
            continue
        print(feature, other_feature)
        if os.path.exists('./%d_out/result_%d'%(seed, feature)):
            results[feature] = np.load('./%d_out/result_%d'%(seed, feature), allow_pickle=True).item()
        else:
            # 
            if feature == other_feature:
                all_other_features = df.columns.difference( [feature] ) # 
                new_other = copy.deepcopy(new_df) # 
                new_other[feature] = df[feature]
                new_other = new_other.to_numpy()
                new_other_yhat = interpreting(new_other)
                new_other_res = eval_model(new_other_yhat[:, 1], target)
                other_feature_performance = new_other_res[M]
                results[feature][feature] = dataset_performance + shuffled_dataset_performance - single_feature_performance[feature] - other_feature_performance
            # 
            elif other_feature in results and feature in results[other_feature] :
                results [feature] [other_feature] = results[other_feature] [feature]
            else:
                # 
                both = copy.deepcopy(df)
                both[feature] = new_df[feature]
                both[other_feature] = new_df[other_feature]
                both = both.to_numpy()
                both_yhat = interpreting(both)
                both_res = eval_model(both_yhat[:, 1], target)
                both_features_shuffled_performance = both_res[M]
                results [feature] [other_feature] = dataset_performance + both_features_shuffled_performance - single_feature_performance[feature] - single_feature_performance [other_feature]
            if len(results[feature]) == 696:
                np.save('./%d_out/result_%d'%(seed, feature), results[feature])
    results_df = pd.DataFrame(results) .T / dataset_performance
    return results_df



asl = np.load('./data/asl.npy')
vbm = np.load('./data/vbm.npy', allow_pickle=True)
data = np.concatenate((vbm, asl), 1)
labels = np.load('./data/labels.npy')
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
    if n_fold == 4:
        break

seed = 1
key_fea = [44,119,134,224,  325,344,323,267,  368,  551,  689,674,675,695]
results_df = fimp(X_test, Y_test, seed, 'acc', key_fea)

results_df.to_csv('permutation_result_seed%s_acc.csv'%(seed))