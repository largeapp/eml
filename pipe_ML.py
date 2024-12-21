import os
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.feature_selection import RFECV
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter

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

    result['tn'] = tn
    result['fp'] = fp
    result['fn'] = fn
    result['tp'] = tp
    return result

def easy_classification(feature, labels, resampling, classify_pipe, parameters, outputdir, seed):
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)
    data = feature
    y = labels
    res = {}
    skf1 = StratifiedKFold(n_splits = 10, shuffle=True, random_state = seed)
    for n_fold,(trains_idx,test_index) in enumerate(skf1.split(data, y)):
        print(f'---------{n_fold}----------')
        X_test, Y_test = data[test_index],y[test_index]
        X_train, Y_train = data[trains_idx], y[trains_idx]
        if resampling:
            rus = RandomUnderSampler(random_state=0,replacement=True)
            X_train,Y_train = rus.fit_resample(X_train,Y_train)
            # print(Counter(Y_train))
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        model = GridSearchCV(classify_pipe, param_grid=parameters, cv=9, verbose=1)
        print(f'train shape:{X_train.shape}, test shape:{X_test.shape}')
        # 
        model.fit(X_train, Y_train)
        model_best = model.best_estimator_
        import pickle
        file = open(outputdir + '/best_model_%02i.pickle' % (n_fold), "wb")
        pickle.dump(model_best, file)
        file.close()
        y_predprob = model_best.predict_proba(X_test)
        res_temp = eval_model(y_predprob[:,1], Y_test)
        for key in res_temp.keys():
            res.setdefault(key, []).append(res_temp[key])
    df = pd.DataFrame(data = res)
    result = df.to_numpy().astype('float')
    mean = np.mean(result, 0)
    std = np.std(result, 0)
    a = []
    for i in range(10):
        temp = '%.4f±%.4f'%(mean[i], std[i])
        a.append(temp)
    df.loc['mean±std'] = a
    return df


def classing_model(feature, labels, clf, parameters, outputdir, seed):
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)
    data = feature
    res = {}
    skf1 = StratifiedKFold(n_splits = 10, shuffle=True, random_state = seed)
    for n_fold, (train_val, test) in enumerate(skf1.split(labels, labels)):
        print('-'*7, n_fold, '-'*7)
        
        train_val_dataset, test_dataset = data[train_val.tolist()], data[test.tolist()]
        train_val_labels = labels[train_val]
        train_val_index = np.arange(len(train_val_dataset))

        train, val, _, _ = train_test_split(train_val_index, train_val_labels, random_state=seed, 
                                            test_size=0.11, shuffle=True, stratify=train_val_labels)
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
        
        # 
        train_val_features = np.concatenate((X_train, X_val),axis=0)
        train_val_labels = np.concatenate((Y_train, Y_val),axis=0)
        test_fold = np.zeros(train_val_features.shape[0])   # Initialize all indices to 0, and 0 represents the validation set for the first round.
        test_fold[:X_train.shape[0]] = -1            # Set the index for the training set to -1, indicating that it will not be included in the validation set.
        from sklearn.model_selection import PredefinedSplit
        ps = PredefinedSplit(test_fold=test_fold)
        model = GridSearchCV(clf, param_grid=parameters, cv=ps, verbose=1)
        print(f'train shape:{X_train.shape}, test shape:{X_test.shape}')
        # model training, grid search
        model.fit(train_val_features, train_val_labels)
        model_best = model.best_estimator_
        import pickle
        file = open(outputdir + '/best_model_%02i.pickle' % (n_fold), "wb")
        pickle.dump(model_best, file)
        file.close()
        y_predprob = model_best.predict_proba(X_test)
        # temp = np.array(model_best.predict_proba(X_train)[:, 1])
        res_temp = eval_model(y_predprob[:,1], Y_test)
        for key in res_temp.keys():
            res.setdefault(key, []).append(res_temp[key])
    df = pd.DataFrame(data = res)
    result = df.to_numpy().astype('float')
    mean = np.mean(result, 0)
    std = np.std(result, 0)
    a = []
    for i in range(10):
        temp = '%.4f±%.4f'%(mean[i], std[i])
        a.append(temp)
    df.loc['mean±std'] = a
    return df

