# -*- coding: utf-8 -*-
"""PGI.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1vkubAaC0pRwPX70Y2kNYua5JSDrGxbQA
"""

import numpy as np

def calculate_pgi(y_true, y_pred, region=0.1):
    """
    Calculate the Partial Gini Index (PGI) for a binary classifier.

    Parameters:
    - y_true: Array-like, true binary labels (0 or 1).
    - y_pred: Array-like, predicted probabilities for the positive class.
    - region: Float, the region of interest as a proportion (e.g., 0.1 for top 10%).

    Returns:
    - PGI score (float).
    """
    # Ensure inputs are NumPy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Sort by predicted probabilities in descending order
    sorted_indices = np.argsort(-y_pred)
    y_true_sorted = y_true[sorted_indices]

    # Define region of interest (top region% of the data)
    n = len(y_true)
    region_count = int(region * n)  # Number of samples in the region
    y_true_region = y_true_sorted[:region_count]

    # Calculate cumulative gain for the region
    cumulative_gain = np.cumsum(y_true_region)

    # Total number of positives in the dataset
    total_positives = np.sum(y_true)

    # Partial Gini Index (sum of cumulative gain normalized by total positives)
    partial_gini = np.sum(cumulative_gain) / total_positives

    # Ideal Gini for the region (if all positives were at the top of the region)
    ideal_gini = np.sum(np.arange(1, region_count + 1)) / total_positives

    # Normalize PGI to scale between 0 and 1
    pgi_score = partial_gini / ideal_gini
    return pgi_score

# # Example usage
# y_true = [0, 1, 1, 0, 1, 0, 1, 0]
# y_pred = [0.1, 0.8, 0.4, 0.6, 0.9, 0.2, 0.7, 0.3]
# region = 0.2  # Top 20% region
#
# pgi_score = calculate_pgi(y_true, y_pred, region=region)
# print("Partial Gini Index (PGI) Score:", pgi_score)