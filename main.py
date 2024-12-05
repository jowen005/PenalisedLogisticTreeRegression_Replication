import importlib

import pandas as pd

import analysis_functions
import numpy as np
importlib.reload(analysis_functions)
from analysis_functions import *
from sklearn.model_selection import  train_test_split
from sklearn.impute import SimpleImputer


# Load the data
dataframe = pd.read_csv("cs-training.csv")
dataframe.rename(columns={'SeriousDlqin2yrs': 'Label'}, inplace=True)
# Fill in missing values by imputing by mean
# Imputation by mean
imp_mean = SimpleImputer(missing_values=pd.NA, strategy='mean')
for col in dataframe.columns:
    if dataframe[col].isnull().to_numpy().any():
        imp_mean.fit(dataframe[col].to_numpy().reshape(-1, 1))
        dataframe[col] = imp_mean.transform(dataframe[col].to_numpy().reshape(-1, 1))

# Separate the labels from the rest of the data
y = dataframe['Label'].to_numpy()
X = dataframe.loc[:, dataframe.columns != 'Label']
# Split dataset evenly and randomly (Nx2 cross validation)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.50, stratify=y)

# Run the model and get the predicted values
y_pred = pltr(X_train, y, X_test)
# Find confusion matrix values using y predicted and y test values
tp, tn, fp, fn = confusion_matrix(y_pred, y_test)