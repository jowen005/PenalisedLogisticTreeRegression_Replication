import numpy as np

# Adapted from AI generated code
def calculate_pgi(y_true, y_prob, region=0.1):
    """
    Calculate the Partial Gini Index (PGI) for a binary classifier.

    Parameters:
    - y_true: Array-like, true binary labels (0 or 1).
    - y_prob: Array-like, predicted probabilities for the positive class.
    - region: Float, the region of interest as a proportion (e.g., 0.1 for top 10%).

    Returns:
    - partial Gini index (float).
    """
    # Ensure inputs are NumPy arrays
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)

    # Sort by predicted probabilities in descending order
    sorted_indices = np.argsort(-y_prob)
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
    pgi = partial_gini / ideal_gini
    return pgi