
import numpy as np
from sklearn.metrics import mean_squared_error, mean_squared_log_error, mean_absolute_error, accuracy_score, r2_score, auc, precision_score, recall_score, f1_score, roc_auc_score, silhouette_score
import sklearn.metrics as skm


def rmse(y, preds):
    return np.sqrt(skm.mean_squared_error(y, preds))


###################################


def rmsle(y, preds):
    if (y < 0).any() or (preds < 0).any():
        return np.inf
    else:
        return np.sqrt(skm.mean_squared_log_error(y, preds))


###################################

def auc_score(y, preds):
    fpr, tpr, thresholds = skm.roc_curve(y, preds)
    return skm.auc(fpr, tpr)
