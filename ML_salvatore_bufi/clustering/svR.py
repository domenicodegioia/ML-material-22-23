import numpy as np
import pandas as pd
from sources.splitting import select_data,get_data_from_indices,get_folds
from sources.utils import get_zscore,apply_zscore
from sklearn import svm
from sklearn.linear_model import LinearRegression
import sources.metrics as skm

''' SVR = SVM PER REGRESSIONE '''

wine = pd.read_csv('../data/wine.csv', header=None)

proc_data = wine.loc[:, wine.columns != 11]

target = wine.loc[:, 11]

#training test splitting
folds_list = get_folds(proc_data, target, kfolds=1, test_size_for_single=0.2)
X_full_train, y_full_train, X_test, y_test = get_data_from_indices(proc_data, target, folds_list[0][0], folds_list[0][1])



#training validation splitting
folds_list = get_folds(X_full_train, y_full_train, kfolds=5)

val_metrics_values = {}

for train_index, val_index in folds_list:
    X_train, y_train, X_val, y_val = get_data_from_indices(X_full_train, y_full_train, train_index, val_index)

    means, stds = get_zscore(X_train.values)
    X_train = apply_zscore(X_train.values, means, stds)
    X_val = apply_zscore(X_val.values, means, stds)

    y_train = y_train.values
    y_val = y_val.values

    # Model 1
    svm_obj = svm.SVR()
    svm_obj.fit(X_train, y_train)
    preds = svm_obj.predict(X_val)
    res = val_metrics_values.get("SVR", {})
    res.setdefault("rmse", []).append(skm.rmse(y_val, preds))
    res.setdefault("mae", []).append(skm.mean_absolute_error(y_val, preds))
    res.setdefault("rmsle", []).append(skm.rmsle(y_val, preds))
    val_metrics_values["SVR"] = res

    # Model 2
    linear_obj = LinearRegression()
    linear_obj.fit(X_train, y_train)
    preds = linear_obj.predict(X_val)
    res = val_metrics_values.get("Linear", {})
    res.setdefault("rmse", []).append(skm.rmse(y_val, preds))
    res.setdefault("mae", []).append(skm.mean_absolute_error(y_val, preds))
    res.setdefault("rmsle", []).append(skm.rmsle(y_val, preds))
    val_metrics_values["Linear"] = res

    # print(np.vstack((y_val.T, preds.T)).T)
    #
    # print(f"Mean Absolute Error:\t\t\t\t\t{skm.mean_absolute_error(y_val, preds)}")
    # print(f"Root Mean Squared Error:\t\t\t\t{skm.rmse(y_val, preds)}")
    # print(f"Root Mean Squared Logarithmic Error:\t{skm.rmsle(y_val, preds)}")

for model, metrics in val_metrics_values.items():
    print(f"\nModel {model} validation results:")
    for metric_name, metric_values in metrics.items():
        print(f"Metric {metric_name}:\t{np.mean(metric_values)}")
    print("\n")


# Generalization Error
print("*** *** ***")
means, stds = get_zscore(X_full_train.values)
X_train = apply_zscore(X_full_train.values, means, stds)
X_test = apply_zscore(X_test.values, means, stds)

test_metrics_values = {}

# Model 1
svm_obj = svm.SVR()
svm_obj.fit(X_train, y_full_train)
preds = svm_obj.predict(X_test)
res = test_metrics_values.get("SVR", {})
res.setdefault("rmse", []).append(skm.rmse(y_test, preds))
res.setdefault("mae", []).append(skm.mean_absolute_error(y_test, preds))
res.setdefault("rmsle", []).append(skm.rmsle(y_test, preds))
test_metrics_values["SVR"] = res

# Model 2
linear_obj = LinearRegression()
linear_obj.fit(X_train, y_full_train)
preds = linear_obj.predict(X=X_test)
res = test_metrics_values.get("Linear", {})
res.setdefault("rmse", []).append(skm.rmse(y_test, preds))
res.setdefault("mae", []).append(skm.mean_absolute_error(y_test, preds))
res.setdefault("rmsle", []).append(skm.rmsle(y_test, preds))
test_metrics_values["Linear"] = res

for model, metrics in test_metrics_values.items():
    print(f"\nModel {model} test results:")
    for metric_name, metric_value in metrics.items():
        print(f"Metric {metric_name}:\t{metric_value[0]}")
    print("\n")


