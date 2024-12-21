import os
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
from pipe_MachineLearning import selector, easy_classification, classing_model

asl = np.load('../data/asl.npy')
# age = np.load('./data/age.npy')
# age = age.reshape(260,1)
# sex = np.load('./data/sex.npy')
# sex = sex.reshape(260,1)
vbm = np.load('../data/vbm.npy', allow_pickle=True)
data = np.concatenate((vbm, asl), 1)


labels = np.load('../data/labels.npy')
outputdir = './vbm+asl/'
subdir = ['svm_linear','svm_rbf','rf','svm_linear1']
seed = 2023

# svm_linear_pipe = Pipeline([
#     ('linear', SVC(kernel='linear', probability=True ,class_weight='balanced'))
# ])
# parameters = {
#     'linear__C':[i for i in range(1,10,1)],
#     'linear__gamma':[0.1,0.01,0.001,0.0001]
# }
# df = easy_classification(data, labels, 'True', svm_linear_pipe, parameters, outputdir+subdir[3], seed)
# df.to_csv('./svm_linear1.csv')
svm_linear =SVC(kernel='linear', probability=True ,class_weight='balanced', random_state=seed)
linear_parameters = {
    'C':[i for i in range(1,10,1)],
    'gamma':[0.1,0.01,0.001,0.0001]
}
sl_result = classing_model(data, labels, svm_linear, linear_parameters, outputdir+subdir[0], seed)
sl_result.to_csv('./svm_linear.csv')
svm_rbf = SVC(kernel='rbf', probability=True ,class_weight='balanced', random_state=seed)
sr_parameters = {
    'C':[i for i in range(1,10,1)],
    'gamma':[0.1,0.01,0.001,0.0001]
}
sr_result = classing_model(data, labels, svm_rbf, sr_parameters, outputdir+subdir[1], seed)
sr_result.to_csv('./svm_rbf.csv')
from sklearn.ensemble import RandomForestClassifier
RF = RandomForestClassifier(class_weight='balanced', random_state=seed)
rf_parameters = {
    'n_estimators':[300, 500]
}
rf_result = classing_model(data, labels, RF, rf_parameters, outputdir+subdir[2], seed)
rf_result.to_csv('./rf.csv')