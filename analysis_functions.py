from os.path import split

from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import  train_test_split
from sklearn.preprocessing import Binarizer
import numpy as np

def pltr(dataframe):
    X = dataframe
    y = dataframe['Label'].to_numpy()

    # Get predictive variables from column names
    predictive_variables = X.columns

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y)
    decision_trees(X_train, y_train)

def decision_trees(X, y, primary, secondary):
    # Use one variable to do the first split
    dt_X1 = X[primary]
    # Use the same variable plus a second one to split the maintained child node
    dt_X2 = X[primary, secondary]
    # Train the 1 split decision tree
    dtree = DecisionTreeClassifier(max_depth=1, max_leaf_nodes=2)
    dtree.fit(dt_X1,y)
    # Get v1 values from 1 split tree
    univariate_threshold, univariate_feature, v1_class = get_split1_info(dtree)
    # Get the retained node to split for the 2 split tree (retiained means to use in analysis not split again)
    # rnode_data = dt_X1[dtree.apply(dt_X1) == retained_id] # How do i get a second variable in?
    # rnode_labels = y[dtree.apply(dt_X1) == retained_id]

    # Train the 2nd split decision tree
    dtree2 = DecisionTreeClassifier(max_depth=2, max_leaf_nodes=3)
    dtree2.fit(dt_X2, y)
    # Get v2 values from 2 split tree
    bivariate_threshold, bivariate_feature, v2_class = get_split2_info(dtree2)

    #univariate_threshold, univariate_feature, v1_class, bivariate_threshold, bivariate_feature, v2_class, v3_class = get_tree_info(dtree)

    # return (univariate_threshold, univariate_feature, v1_class, bivariate_threshold, bivariate_feature, v2_class, v3_class)

# Extract the thresholds, features, and classes for each split and leaf nodes in 1 split tree
def get_split1_info(fitted_tree):
    n_nodes = fitted_tree.tree_.node_count
    children_left = fitted_tree.tree_.children_left
    children_right = fitted_tree.tree_.children_right
    feature = fitted_tree.tree_.feature
    threshold = fitted_tree.tree_.threshold
    values = fitted_tree.tree_.value

    # node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
    # is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    # Flag that turns false if we store a class for a node on depth 2
    node_stack = [(0, 0)]

    while len(node_stack) > 0:
        node_id, depth = node_stack.pop()
        # If the left and right child of a node is not the same it is split node
        is_split_node = children_left[node_id] != children_right[node_id]
        # If a split node, append left and right children and depth to `stack' to visit them next
        if is_split_node:
            node_stack.append((children_left[node_id], depth + 1))
            node_stack.append((children_right[node_id], depth + 1))
            v1_node = children_right[node_id]
            # The root split gives the univariate threshold and feature
            if depth == 0:
                univariate_threshold = threshold[node_id]
                univariate_feature = feature[node_id]
            else:
                print('there should be no split at depth 1 or lower for 1 split tree')
        else:
            # is_leaves[node_id] = True
            leaf_values = values[node_id]
            if depth == 1:
                if n_nodes[node_id] == v1_node:
                    v1_class = np.argmax(leaf_values)
                else:
                    # This is the left child so don't get the class from it.
                    continue
            else:
                print('there should not be any leaves at depth 2 or lower for 1 split tree')

    # Return values
    return (univariate_threshold, univariate_feature, v1_class)

# Extract the thresholds, features, and classes for each split and leaf nodes in 2 split tree
def get_split2_info(fitted_tree):
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
            # if depth == 0:
            #     univariate_threshold = threshold[node_id]
            #     univariate_feature = feature[node_id]
            # The split at depth 1 gives the bivariate threshold and feature
            if depth == 1:
                bivariate_threshold = threshold[node_id]
                bivariate_feature = feature[node_id]
            elif depth >= 2:
                print('there should be no split at depth 2 or lower')
        else:
            # is_leaves[node_id] = True
            leaf_values = values[node_id]
            # if depth == 1:
            #     v1_class = np.argmax(leaf_values)
            if depth == 2:
                # If we haven't visited a node at depth 2 before then skip this node's class
                if first_d2_leaf:
                    # v2_class = np.argmax(leaf_values)
                    first_d2_leaf = False
                    continue
                # The second time we visit a node at depth 2 this node's class is v2
                else:
                    v2_class = np.argmax(leaf_values)
                    # v3_class = np.argmax(leaf_values)
            elif depth >= 3:
                print('there should not be any leaves at depth 3 or lower')

    # Return values
    # return(univariate_threshold, univariate_feature, v1_class, bivariate_threshold, bivariate_feature,
    #        v2_class, v3_class)
    return (bivariate_threshold, bivariate_feature, v2_class)
