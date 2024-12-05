from os.path import split

import pandas as pd
from numpy.ma.core import argmax
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import  train_test_split
from sklearn.preprocessing import Binarizer
import numpy as np
import matlab.engine
import AdaptoLogit as al

# Returns fit plt regression model
def pltr(X, y, X_test):
    # Get predictive variables from column names
    predictive_variables = X.columns

    # Get the univariate and bivariate effects without duplicates
    univariate_effects, bivariate_effects = decision_trees(X, y)

    # Apply threshold effects to dataset
    X_effects_train = apply_effects(univariate_effects, bivariate_effects, X)

    # Perform the lasso to get relevant variables for the logistic regression and perform logistic regression
    # Start the matlab engine since the functions used in the paper are from matlab
    eng = matlab.engine.start_matlab("-desktop")
    # Fit the X_train and y_train in 10 fold cross validated adaptive lasso penalized logistic regression
    model = eng.glm_logistic(y, X_effects_train,  eng.workspace["nointercept"])
    weights = al.AdaptiveWeights(weight_technique='ridge') # Just used default power weights may change to 1 for v in paper
    cv_fit = eng.cv_penalized(model, eng.workspace['@p_adaptive'],  eng.workspace["gamma"], 0.5,
                              eng.workspace["adaptivewt"], {weights},  eng.workspace["folds"], 10)

    y_pred = eng.predict(cv_fit, X_test)
    # Stop the matlab engine
    eng.quit()

    # Return the predicted target values
    return y_pred



def decision_trees(X, y):
    # Create 3D arrays storing each threshold with its feature and class
    # Univariate Array
    univariate_effects = []
    bivariate_effects = []
    # Get predictive variables from column names
    for primary in X.columns:
        # Use one variable to do the first split
        dt_X1 = X[primary]
        for secondary in X.columns:
            # Use the same variable plus a second one to split the child node
            dt_X2 = X[primary, secondary]
    # Use one variable to do the first split
    # dt_X1 = X[primary]
    # Use the same variable plus a second one to split the maintained child node
    # dt_X2 = X[primary, secondary]
             # Train the 1 split decision tree
            dtree = DecisionTreeClassifier(max_depth=1, max_leaf_nodes=2)
            dtree.fit(dt_X1,y)
            # Get v1 values from 1 split tree
            univariate_threshold, univariate_feature, v1_class_left, v1_class_right = get_split1_info(dtree)
            univariate_effects.append([univariate_feature,univariate_threshold, v1_class_left, v1_class_right])
            # Get the retained node to split for the 2 split tree (retiained means to use in analysis not split again)
            # rnode_data = dt_X1[dtree.apply(dt_X1) == retained_id] # How do i get a second variable in?
            # rnode_labels = y[dtree.apply(dt_X1) == retained_id]

            # Train the 2nd split decision tree
            dtree2 = DecisionTreeClassifier(max_depth=2, max_leaf_nodes=3)
            dtree2.fit(dt_X2, y)
            # Get v2 values from 2 split tree
            (bivariate_threshold1, bivariate_feature1, bivariate_threshold2, bivariate_feature2,
             v2_class_left, v2_class_right, thres_1) = get_split2_info(dtree2)
            # Only add if this is not a duplicate bivariate effect
            bivariate_effect = [bivariate_threshold1, bivariate_feature1, bivariate_threshold2,
                                bivariate_feature2, v2_class_left, v2_class_right, thres_1]
            if bivariate_effect not in bivariate_effects:
                bivariate_effects.append(bivariate_effect)

    #univariate_threshold, univariate_feature, v1_class, bivariate_threshold, bivariate_feature, v2_class, v3_class = get_tree_info(dtree)

    # return (univariate_threshold, univariate_feature, v1_class, bivariate_threshold, bivariate_feature, v2_class, v3_class)

    # Return the univariate and bivariate effects
    return univariate_effects, bivariate_effects

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
            v1_node = children_left[node_id]
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
                    # This is the left child and the predicted value when the threshold is true
                    v1_class_left = np.argmax(leaf_values)
                else:
                    # This is the right child and the predicted value when the threshold is false
                    v1_class_right = np.argmax(leaf_values)
                    continue
            else:
                print('there should not be any leaves at depth 2 or lower for 1 split tree')

    # Return values
    return univariate_threshold, univariate_feature, v1_class_left, v1_class_right

# Extract the thresholds, features, and classes for each split and leaf nodes in 2 split tree
def get_split2_info(fitted_tree):
    n_nodes = fitted_tree.tree_.node_count
    children_left = fitted_tree.tree_.children_left
    children_right = fitted_tree.tree_.children_right
    feature = fitted_tree.tree_.feature
    threshold = fitted_tree.tree_.threshold
    values = fitted_tree.tree_.value

    # node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
    # is_leaves = np.zeros(shape=n_nodes, dtype=bool)
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
            # The root split gives part 1 of the bivariate threshold and feature
            # We're using the 0 child so if the value is <= threshold then we check for part 2
            if depth == 0:
                bivariate_threshold1 = threshold[node_id]
                bivariate_feature1 = feature[node_id]
                l_child = children_left[node_id]
                r_child = children_right[node_id]
            # The split at depth 1 gives the bivariate threshold and feature
            if depth == 1:
                bivariate_threshold2 = threshold[node_id]
                bivariate_feature2 = feature[node_id]
                v2_node = children_left[node_id]
                if n_nodes[node_id] == l_child:
                    thres_1 = True
                elif n_nodes[node_id] == r_child:
                    thres_1 = False
            elif depth >= 2:
                print('there should be no split at depth 2 or lower')
        else:
            # is_leaves[node_id] = True
            leaf_values = values[node_id]
            # if depth == 1:
            #     v1_class = np.argmax(leaf_values)
            if depth == 2:
                if n_nodes[node_id] == v2_node:
                    #  This is the left child and the predicted value when the threshold is true
                    v2_class_left = np.argmax(leaf_values)
                # If we haven't visited a node at depth 2 before then this node's class is the class if the threshold is
                # not met
                # if first_d2_leaf:
                #     v2_not_class = np.argmax(leaf_values)
                #     first_d2_leaf = False
                #     continue
                else:
                    # This is the right child and the predicted value when the threshold is false
                    v2_class_right = np.argmax(leaf_values)
                    # v3_class = np.argmax(leaf_values)
            elif depth >= 3:
                print('there should not be any leaves at depth 3 or lower')

    # Return values
    # return(univariate_threshold, univariate_feature, v1_class, bivariate_threshold, bivariate_feature,
    #        v2_class, v3_class)
    return (bivariate_threshold1, bivariate_feature1, bivariate_threshold2, bivariate_feature2,
            v2_class_left, v2_class_right, thres_1)

# Replace the variables with the new threshold variables and return that new array
def apply_effects(univariate, bivariate, X):
    predictive_variables = X.columns
    # Create new dataset
    X_new = pd.DataFrame()
    for u in univariate:
        feature = u[1]
        threshold = u[0]
        effect_value_left = u[2]
        effect_value_right = u[3]
        applied_effects = []
        # Add header similar to table 3 in paper for column
        feature_name = predictive_variables[feature]
        # Determine which way the inequality goes to match with the predicted values
        if effect_value_left == 1:
            decision_rule = feature_name + ' <= ' + str(threshold)
        else:
            decision_rule = feature_name + ' > ' + str(threshold)
        # Go through the dataset and determine what each individual would be predicted based on the rule
        for index, row in X.iterrows():
            if row[feature_name] <=  threshold:
                applied_effects[index] =  effect_value_left
            else:
                applied_effects[index] = effect_value_right
        # Add column for decision rule into dataframe
        X_new[decision_rule] = applied_effects

        # Apply threshold to column from original dataset
        # X[:, feature]
        # Copy that column into new dataset

    for b in bivariate:
        feature1 = b[1]
        feature2 = b[3]
        threshold1 = b[0]
        threshold2= b[2]
        effect_value_left = b[4]
        effect_value_right = b[5]
        thres_1 = b[6]
        applied_effects = []
        # Add header similar to table 3 in paper for column
        feature1_name = predictive_variables[feature1]
        feature2_name = predictive_variables[feature2]
        # Determine the inequality direction for the first part of the decision rule
        if thres_1:
            dr_p1 = feature1_name + ' <= ' + str(threshold1) + ' & '
        else:
            dr_p1 = feature1_name + ' > ' + str(threshold1) + ' & '
        # Determine the inequality direction for the second part of the decision rule
        if effect_value_left == 1:
            dr_p2 = feature2_name + ' <= ' + str(threshold2)
        else:
            dr_p2 = feature2_name + ' > ' + str(threshold2)

        # Combine the parts to make the decision rule
        decision_rule = dr_p1 + dr_p2

        # Go through the dataset and determine what each individual would be predicted based on the rule
        for index, row in X.itterows():
            if thres_1:
                if row[feature1_name] <= threshold1:
                    if row[feature2_name] <= threshold2:
                        applied_effects[index] = effect_value_left
                    else:
                        applied_effects[index] = effect_value_right
                else:
                    applied_effects[index] = 0
            else:
                if row[feature1_name] > threshold1:
                    if row[feature2_name] <= threshold2:
                        applied_effects[index] = effect_value_left
                    else:
                        applied_effects[index] = effect_value_right
                else:
                    applied_effects[index] = 0

        # Add column for decision rule into dataframe
        X_new[decision_rule] = applied_effects


    # Return the changed X
    return X_new