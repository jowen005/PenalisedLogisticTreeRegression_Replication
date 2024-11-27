from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from sklearn.preprocessing import Binarizer
import numpy as np

def decision_trees(X, primary, secondary, y):
    # Only use the two passed columns to create the decision tree
    dt_X = X[primary, secondary]

    dtree = DecisionTreeClassifier(max_depth=2)
    dtree.fit(dt_X,y)

