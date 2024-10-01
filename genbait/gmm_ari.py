import pandas as pd
import numpy as np
import random
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture

def get_gmm_hard_assignments(data, n_clusters, seed):
    """
    Fit a Gaussian Mixture Model (GMM) to the input data and return hard cluster assignments.

    Args:
    - data (numpy.ndarray): The input data matrix where each row is a sample.
    - n_clusters (int): The number of clusters (components) for the GMM.
    - seed (int): Random seed for reproducibility.

    Returns:
    - numpy.ndarray: The hard cluster assignments for each sample in the data.
    """
    # Initialize the Gaussian Mixture Model with the specified number of clusters and seed
    gmm = GaussianMixture(n_components=n_clusters, random_state=seed)
    
    # Fit the GMM model to the data
    gmm.fit(data)
    
    # Return the hard assignments (most likely cluster for each sample)
    return gmm.predict(data)

def precalculate_gmm_assignments_for_original(df_original, cluster_numbers, seed):
    """
    Precompute GMM hard cluster assignments for the original dataset for a range of cluster numbers.

    Args:
    - df_original (pd.DataFrame): The preprocessed data for the original dataset.
    - cluster_numbers (list): List of cluster numbers to compute GMM cluster assignments for.
    - seed (int): Random seed for reproducibility.

    Returns:
    - dict: A dictionary where each key is a cluster number, and the value is the GMM hard assignments.
    """
    # Transpose the original dataset so that rows correspond to samples
    data = df_original.transpose().values
    
    # Dictionary to store the GMM assignments for different cluster numbers
    original_assignments = {}
    
    # Loop through each cluster number
    for n_clusters in cluster_numbers:
        # Get the hard cluster assignments for the original dataset with the given number of clusters
        original_assignments[n_clusters] = get_gmm_hard_assignments(data, n_clusters, seed)
    
    return original_assignments

def calculate_ari_between_assignments(original_assignments, df_original, df_subset, n_clusters, seed):
    """
    Calculate the Adjusted Rand Index (ARI) between GMM cluster assignments of the original and subset data.

    Args:
    - original_assignments (dict): Precomputed GMM assignments for the original dataset.
    - df_original (pd.DataFrame): The preprocessed data for the original dataset.
    - df_subset (pd.DataFrame): The subset of the data for which GMM clustering is to be compared.
    - n_clusters (int): The number of clusters (components) for the GMM.
    - seed (int): Random seed for reproducibility.

    Returns:
    - float: The ARI score between the original and subset GMM assignments.
    """
    # Identify the common preys (columns) between the original and subset data
    common_preys = df_original.columns.intersection(df_subset.columns)
    
    # Transpose the subset data and select only the common preys
    df_subset_common = df_subset[common_preys].transpose().values
    
    # Get the GMM hard assignments for the subset data
    subset_assignments = get_gmm_hard_assignments(df_subset_common, n_clusters, seed)
    
    # Get the GMM assignments for the original data corresponding to the common preys
    common_original_assignments = original_assignments[n_clusters][np.isin(df_original.columns, common_preys)]
    
    # Calculate and return the ARI score between the original and subset cluster assignments
    return adjusted_rand_score(common_original_assignments, subset_assignments)

def calculate_gmm_ari(df_norm, selected_baits, cluster_numbers, seed=4):
    """
    Calculate the Adjusted Rand Index (ARI) for GMM clustering between original and selected baits data.

    Args:
    - df_norm (pd.DataFrame): The preprocessed data.
    - selected_baits (list): List of selected baits to analyze.
    - cluster_numbers (list): List of cluster numbers to fit the GMM.
    - seed (int): Random seed for reproducibility (default is 4).

    Returns:
    - dict: A dictionary with cluster numbers as keys and ARI values as values.
    """
    # Set random seed for reproducibility
    random.seed(seed)
    
    # Subset the data to include only the selected baits
    df_subset = df_norm.loc[selected_baits]
    
    # Mask to remove columns (preys) that have all zero values
    mask = (df_subset != 0).any(axis=0)
    df_subset = df_subset.loc[:, mask]
    
    # Precompute GMM hard assignments for the original dataset for all cluster numbers
    original_assignments = precalculate_gmm_assignments_for_original(df_norm, cluster_numbers, seed)
    
    # Dictionary to store the ARI results for each cluster number
    results = {}
    
    # Loop through each cluster number and calculate the ARI between the original and subset data
    for cluster_number in cluster_numbers:
        ari = calculate_ari_between_assignments(original_assignments, df_norm, df_subset, cluster_number, seed)
        results[cluster_number] = ari
    
    return results
