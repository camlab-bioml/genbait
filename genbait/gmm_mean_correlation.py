import pandas as pd
import numpy as np
import random
from sklearn.mixture import GaussianMixture
from scipy.optimize import linear_sum_assignment

def get_gmm_soft_assignments(data, n_clusters, seed):
    """
    Fit a Gaussian Mixture Model (GMM) to the input data and return soft cluster assignments (probabilities).

    Args:
    - data (numpy.ndarray): The input data matrix where each row is a sample.
    - n_clusters (int): The number of clusters (components) for the GMM.
    - seed (int): Random seed for reproducibility.

    Returns:
    - numpy.ndarray: The soft cluster assignments (probability distributions) for each sample.
    """
    # Initialize the Gaussian Mixture Model with the specified number of clusters and random seed
    gmm = GaussianMixture(n_components=n_clusters, random_state=seed)
    
    # Fit the GMM model to the data
    gmm.fit(data)
    
    # Return the soft assignments (probability distribution over clusters for each sample)
    return gmm.predict_proba(data)

def average_cluster_correlation(original_probs, subset_probs):
    """
    Calculate the average cluster correlation between original and subset soft cluster assignments.

    Args:
    - original_probs (numpy.ndarray): Soft cluster assignments (probabilities) from the original dataset.
    - subset_probs (numpy.ndarray): Soft cluster assignments (probabilities) from the subset dataset.

    Returns:
    - float: The mean correlation value between original and subset clusters.
    """
    # Calculate the correlation matrix between original and subset cluster assignments
    corr_matrix = np.dot(original_probs.T, subset_probs)
    
    # Compute the cost matrix as 1 - correlation matrix for alignment
    cost_matrix = 1 - corr_matrix
    
    # Apply the Hungarian algorithm to match clusters in the original and subset data
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    # Reorder the subset probabilities to match the original clusters
    subset_probs_reordered = subset_probs[:, col_ind]
    
    # Compute the correlation between the original and reordered subset probabilities
    reordered_corr_matrix = np.corrcoef(original_probs.T, subset_probs_reordered.T)[:original_probs.shape[1], original_probs.shape[1]:]
    
    # Extract the diagonal of the correlation matrix (correlations of aligned clusters)
    diag = np.diag(reordered_corr_matrix)
    
    # Return the mean of the diagonal correlations
    return np.mean(diag)

def precompute_original_soft_assignments(df_norm, cluster_numbers, seed):
    """
    Precompute GMM soft cluster assignments for the original dataset for various cluster numbers.

    Args:
    - df_norm (pd.DataFrame): The preprocessed data for the original dataset.
    - cluster_numbers (list): List of cluster numbers to compute GMM soft assignments for.
    - seed (int): Random seed for reproducibility.

    Returns:
    - dict: A dictionary where each key is a cluster number, and the value is the soft assignments (probabilities).
    """
    # Convert the original dataset to a numpy array (transpose so that rows correspond to samples)
    original_data_np = df_norm.T.to_numpy()
    
    # Dictionary to store the GMM soft assignments for different cluster numbers
    original_probs_dict = {cluster_number: get_gmm_soft_assignments(original_data_np, cluster_number, seed) for cluster_number in cluster_numbers}
    
    return original_probs_dict

def calculate_gmm_mean_correlation(df_norm, selected_baits, cluster_numbers, seed=4):
    """
    Calculate the mean correlation between GMM soft cluster assignments for original and selected baits data.

    Args:
    - df_norm (pd.DataFrame): The preprocessed data.
    - selected_baits (list): List of selected baits to analyze.
    - cluster_numbers (list): List of cluster numbers to fit the GMM.
    - seed (int): Random seed for reproducibility.

    Returns:
    - dict: A dictionary with cluster numbers as keys and mean correlation values as values.
    """
    # Set random seed for reproducibility
    random.seed(seed)
    
    # Subset the data to include only the selected baits and transpose to have samples as rows
    subset_data_np = df_norm.loc[selected_baits].T.to_numpy()
    
    # Precompute soft cluster assignments for the original dataset for all specified cluster numbers
    original_probs_dict = precompute_original_soft_assignments(df_norm, cluster_numbers, seed)
    
    # Dictionary to store the mean correlation results for each cluster number
    results = {}
    
    # Loop through each cluster number and calculate the mean correlation between original and subset data
    for cluster_number in cluster_numbers:
        # Get the soft assignments (probabilities) for the original and subset data
        original_probs = original_probs_dict[cluster_number]
        subset_probs = get_gmm_soft_assignments(subset_data_np, cluster_number, seed)
        
        # Calculate the average cluster correlation between the original and subset soft assignments
        mean_corr = average_cluster_correlation(original_probs, subset_probs)
        
        # Store the result in the dictionary
        results[cluster_number] = mean_corr
    
    return results
