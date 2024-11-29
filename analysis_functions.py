from os.path import split

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

def decision_trees(X, primary, y, secondary = None):
    # Only use the two passed columns to create the decision tree
    dt_X = X[primary, secondary]

    dtree = DecisionTreeClassifier(max_depth=2, max_leaf_nodes=3)
    dtree.fit(dt_X,y)

    univariate_threshold, univariate_feature, v1_class, bivariate_threshold, bivariate_feature, v2_class, v3_class = get_tree_info(dtree)

    return (univariate_threshold, univariate_feature, v1_class, bivariate_threshold, bivariate_feature, v2_class, v3_class)

# Extract the thresholds, features, and clasees for each split and leaf nodes
def get_tree_info(fitted_tree):
    n_nodes = fitted_tree.tree_.node_count
    children_left = fitted_tree.tree_.children_left
    children_right = fitted_tree.tree_.children_right
    feature = fitted_tree.tree_.feature
    threshold = fitted_tree.tree_.threshold
    values = fitted_tree.tree_.value

    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    # Flag that turns false if we store a class for a node on depth 2
    first_d2_leaf = True
    node_stack = [(0, 0)]

    while len(node_stack) > 0:
        node_id, depth = node_stack.pop()
        # If the left and right child of a node is not the same it is split node
        is_split_node = children_left[node_id] != children_right[node_id]
        # If a split node, append left and right children and depth to `stack' to visit them next
        if is_split_node:
            node_stack.append((children_left[node_id], depth + 1))
            node_stack.append((children_right[node_id], depth + 1))
            # The root split gives the univariate threshold and feature
            if depth == 0:
                univariate_threshold = threshold[node_id]
                univariate_feature = feature[node_id]
            # The split at depth 1 gives the bivariate threshold and feature
            elif depth == 1:
                bivariate_threshold = threshold[node_id]
                bivariate_feature = feature[node_id]
            else:
                print('there should be no split at depth 2 or lower')
        else:
            is_leaves[node_id] = True
            leaf_values = values[node_id]
            if depth == 1:
                v1_class = np.argmax(leaf_values)
            elif depth == 2:
                # If we haven't visited a node at depth 2 before then this node's class is v2
                if first_d2_leaf:
                    v2_class = np.argmax(leaf_values)
                    first_d2_leaf = False
                # The second time we visit a node at depth 2 this node's class is v3
                else:
                    v3_class = np.argmax(leaf_values)
            else:
                print('there should not be any leaves at lower depths')

    # Return values
    return(univariate_threshold, univariate_feature, v1_class, bivariate_threshold, bivariate_feature,
           v2_class, v3_class)
