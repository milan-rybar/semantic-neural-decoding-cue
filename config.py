import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

from datasets import Dataset1, Dataset2

RESULTS_PATH = '/home/milan/eeg_cue_paper_results'

DATASETS = {
    'Data1': dict(cls=Dataset1, path='/home/milan/dataset_PS1'),
    'Data2': dict(cls=Dataset2, path='/home/milan/dataset_PS2')
}

CLASSIFIERS_OPTIONS = {
    'LR_L1': LogisticRegression(max_iter=100000, penalty='l1', solver='liblinear'),
    'LR_L1_CV': LogisticRegressionCV(Cs=10, cv=10, max_iter=100000, penalty='l1', solver='saga', n_jobs=4),
    'SVM_C1': SVC(kernel='rbf', C=1, cache_size=10000),
    'SVM_CV_C': GridSearchCV(SVC(kernel='rbf', cache_size=10000), {'C': np.logspace(-2, 2, 100)}, cv=10, n_jobs=4),
    'LDA': LinearDiscriminantAnalysis(),
    'LR_L2': LogisticRegression(max_iter=100000, penalty='l2')
}
