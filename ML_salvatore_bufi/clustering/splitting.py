
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.model_selection import ShuffleSplit


def get_folds(proc_data, target, kfolds=1, stratified=False, test_size_for_single=0.2, random_subsampling_folds=1):
    if kfolds != 1:
        if stratified:
            skf = StratifiedKFold(n_splits=kfolds, random_state=42, shuffle=True)
        else:
            skf = KFold(n_splits=kfolds, random_state=42, shuffle=True)

        folds = [(train_index, test_index) for train_index, test_index in
                 skf.split(proc_data, target)]
    else:
        if stratified:
            ss = StratifiedShuffleSplit(n_splits=random_subsampling_folds, test_size=test_size_for_single, random_state=42)
        else:
            ss = ShuffleSplit(n_splits=random_subsampling_folds, test_size=test_size_for_single, random_state=42)

        folds = [(train_index, test_index) for train_index, test_index in
                 ss.split(proc_data, target)]
    return folds


def get_data_from_indices(proc_data, target, train_indices, test_indices):
    # per y potremmo farlo senza iloc es: y_train = proc_data.[train_indices]
    X_train = proc_data.iloc[train_indices]
    y_train = target.iloc[train_indices]
    X_test = proc_data.iloc[test_indices]
    y_test = target.iloc[test_indices]
    return X_train.reset_index(drop=True), y_train.reset_index(drop=True), X_test.reset_index(drop=True), y_test.reset_index(drop=True)


def select_data(proc_data, features):
    return pd.DataFrame(proc_data, columns=features).values
