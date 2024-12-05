import importlib
import analysis_functions
importlib.reload(analysis_functions)
from analysis_functions import *
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import  train_test_split
from sklearn.preprocessing import Binarizer
import numpy as np
import matlab.engine
import AdaptoLogit as al

# Load the data
dataframe = pd.read_csv("cs-test.csv")
# Fill in missing values by imputing by mean

# Standardize the data

# Separate the labels from the rest of the data
y = dataframe['Label'].to_numpy()
X = dataframe.loc[:, dataframe.columns != 'Label']
# Split dataset evenly and randomly (Nx2 cross validation)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.50, stratify=y)

# Run the model and get the predicted values
y_pred = pltr(X_train, y, X_test)