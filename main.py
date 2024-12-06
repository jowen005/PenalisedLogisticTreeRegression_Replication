import importlib
from operator import index

import pandas as pd

import analysis_functions
import numpy as np


importlib.reload(analysis_functions)
from analysis_functions import *
import PGI
importlib.reload(PGI)
from PGI import *
import ks_score
importlib.reload(ks_score)
from ks_score import *
from sklearn.model_selection import  train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix, roc_auc_score, brier_score_loss
from scipy.stats import kstest



# Load the data
dataframe = pd.read_csv("cs-training.csv", index_col=False)
# d2 = pd.read_excel("default of credit card clients(Taiwan).xls"), index_col=None)
# Remove index column that comes with dataset
dataframe.drop('Unnamed: 0', axis='columns', inplace=True)
dataframe.rename(columns={'SeriousDlqin2yrs': 'Label'}, inplace=True)

# non_rounded_variables = ['RevolvingUtilizationOfUnsecuredLines', 'DebtRatio', 'MonthlyIncome']
# Fill in missing values by imputing by mean
# Imputation by mean if we need to round the mean
# for col in dataframe.columns:
#     # Do not round the mean for certain variables
#     if col in non_rounded_variables:
#         mean_var = dataframe[col].mean()
#     # Round the mean for all other variables
#     else:
#         mean_var = dataframe[col].mean().round()
#     dataframe[col] = dataframe[col].fillna(mean_var)
# Imputation by mean if we don't need to round the mean
imp_mean = SimpleImputer(missing_values=pd.NA, strategy='mean')
for col in dataframe.columns:
    if dataframe[col].isnull().to_numpy().any():
        imp_mean.fit(dataframe[col].to_numpy().reshape(-1, 1))
        dataframe[col] = imp_mean.transform(dataframe[col].to_numpy().reshape(-1, 1))

# Separate the labels from the rest of the data
y = dataframe['Label'].to_numpy()
X = dataframe.loc[:, dataframe.columns != 'Label']
# Split dataset evenly and randomly (Nx2 cross validation)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.50, stratify=y)

# Run the model and get the predicted values
# y_pred = pltr(X_train, y_train, X_test)
y_pred, y_test, y_prob, decision_set = pltr(X, y)

# Find confusion matrix values using y predicted and y test values
# tp, tn, fp, fn = confusion_matrix(y_pred, y_test).ravel()
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
# Proportion of Correct Classification
pcc = (tp + tn) / (tp + tn + fp + fn)
# AUC Score
auc_score = roc_auc_score(y_test, y_pred)
# Brier score
brier_score = brier_score_loss(y_test, y_prob, pos_label=1)

# Calculate KS Statistic
# # Get the probabilities of the positive class in descending order for the positive class
# y_prob_descend = sorted(y_prob, reverse=True)
# positive_values = y_prob_descend[y_test == 1]
# negative_values = y_prob_descend[y_test == 0]
#
# k_score = max([abs(np.mean(positive_values <= threshold) - np.mean(negative_values <= threshold)) for threshold in y_prob])
k_score = calculate_ks_statistic(y_test, y_pred)
partial_gini_index = calculate_pgi(y_test, y_pred)

# Replicate table 1 and 4
pltr_data = {'AUC': auc_score, 'PGI': partial_gini_index, 'PCC': pcc, 'KS': k_score, 'BS': brier_score}
pltr_performance = pd.DataFrame(pltr_data)
pltr_performance.to_excel('tab1.xlsx', index=False)
# Replicate table 2 and 5
pltr_i_data = {'Size of the decision set': decision_set, 'Maximal number of predicates': 2}
pltr_interp = pd.DataFrame(pltr_i_data)
pltr_interp.to_excel('tab2.xlsx', index=False)


