from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import  train_test_split
from sklearn.preprocessing import Binarizer
import numpy as np

def pltr(dataframe):
    X = dataframe
    y = dataframe['Label'].to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y)
    decision_trees(X_train, y_train)

def decision_trees(X, primary, secondary, y):
    # Only use the two passed columns to create the decision tree
    dt_X = X[primary, secondary]

    dtree = DecisionTreeClassifier(max_depth=2)
    dtree.fit(dt_X,y)



