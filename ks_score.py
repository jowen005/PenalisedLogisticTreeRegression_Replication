# -*- coding: utf-8 -*-

# Adapted from AI generated code
import numpy as np

def calculate_ks_statistic(y_true, y_prob):
    """
    Calculate the Kolmogorov-Smirnov (KS) statistic for binary classification.

    Parameters:
    - y_true: List or array of true binary labels (0 or 1).
    - y_prob: List or array of predicted probabilities for the positive class.

    Returns:
    - KS statistic (float).
    """
    # Ensure the inputs are NumPy arrays
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)

    # Sort by predicted probabilities (descending order)
    sorted_indices = np.argsort(-y_prob)
    y_true_sorted = y_true[sorted_indices]

    # Total positives and negatives
    total_positives = np.sum(y_true)
    total_negatives = len(y_true) - total_positives

    # Cumulative counts
    cum_positives = np.cumsum(y_true_sorted) / total_positives
    cum_negatives = np.cumsum(1 - y_true_sorted) / total_negatives

    # KS statistic: Maximum difference between cumulative distributions
    ks_statistic = np.max(np.abs(cum_positives - cum_negatives))
    return ks_statistic

# # Example usage
# y_true = [0, 1, 0, 1, 1, 0, 1, 0]
# y_prob = [0.1, 0.8, 0.4, 0.6, 0.9, 0.2, 0.7, 0.3]
# ks_stat = calculate_ks_statistic(y_true, y_prob)
# print("KS Statistic:", ks_stat)
#
# from scipy.stats import ks_2samp
#
# # Example distributions
# group1 = [0.1, 0.4, 0.35, 0.8]
# group2 = [0.05, 0.3, 0.4, 0.76]
#
# # KS statistic and p-value
# ks_stat, p_value = ks_2samp(group1, group2)
# print("KS Statistic:", ks_stat)
# print("P-value:", p_value)