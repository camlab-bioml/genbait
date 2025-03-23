import pandas as pd
import numpy as np
from sklearn.decomposition import NMF
from scipy.optimize import linear_sum_assignment

def kl_divergence(P, Q):
    """
    Compute the Kullback-Leibler divergence between two probability distributions P and Q.
    
    Args:
    - P (np.ndarray): Probability distribution P.
    - Q (np.ndarray): Probability distribution Q.

    Returns:
    - float: Kullback-Leibler divergence between P and Q.
    """
    return np.sum(np.where(P != 0, P * np.log(P / Q), 0), axis=0)

def calculate_nmf_kl_divergence(df_norm, selected_baits, n_components):
    """
    Calculates the NMF KL divergence for the selected baits.

    Args:
    - df_norm (pd.DataFrame): The preprocessed data.
    - selected_baits (list): List of selected baits.
    - n_components (int): The number of NMF components.

    Returns:
    - diagonal_mean (float): The mean KL divergence across all components.
    - diagonal (np.ndarray): KL divergence values for each component.
    """
    
    # Subset the data using the selected baits
    original_data = df_norm.to_numpy()
    
    # Get indices of the selected baits
    subset_indices = list(df_norm.index.get_indexer(selected_baits))
    subset_data = original_data[subset_indices, :]

    # Check if the number of selected baits is valid (should be greater than n_components)
    if len(subset_indices) <= n_components:
        return None, "Number of selected baits is less than or equal to the number of components."

    # Apply NMF to the original dataset
    nmf = NMF(n_components=n_components, init='nndsvd', l1_ratio=1, random_state=46)
    scores_matrix_original = nmf.fit_transform(original_data)
    basis_matrix_original = nmf.components_.T

    # Apply NMF to the subset of selected baits
    scores_matrix_subset = nmf.fit_transform(subset_data)
    basis_matrix_subset = nmf.components_.T

    # Calculate cosine similarity matrix and apply Hungarian algorithm to match components
    cosine_similarity_matrix = np.dot(basis_matrix_original.T, basis_matrix_subset)
    cost_matrix = 1 - cosine_similarity_matrix
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    basis_matrix_subset_reordered = basis_matrix_subset[:, col_ind]

    # Normalize the matrices with a small epsilon value to avoid division by zero
    epsilon = 1e-10  # Small constant to avoid numerical instability
    basis_matrix_original_normalized = (basis_matrix_original + epsilon) / np.sum(basis_matrix_original + epsilon, axis=0)
    basis_matrix_subset_reordered_normalized = (basis_matrix_subset_reordered + epsilon) / np.sum(basis_matrix_subset_reordered + epsilon, axis=0)

    # Initialize the KL divergence matrix
    kl_div_matrix = np.zeros((n_components, n_components))

    # Calculate KL divergence between each pair of components
    for i in range(n_components):
        for j in range(n_components):
            kl_div_matrix[i, j] = kl_divergence(basis_matrix_original_normalized[:, i], basis_matrix_subset_reordered_normalized[:, j])

    # Extract the diagonal of the KL divergence matrix (corresponding component pairs)
    diagonal = np.diag(kl_div_matrix)
    
    # Calculate the mean KL divergence from the diagonal values
    diagonal_mean = np.mean(diagonal)

    # Calculate the min KL divergence from the diagonal values
    diagonal_min = np.max(diagonal)

    # Return the mean KL divergence and the KL divergence values for each component
    return diagonal_mean, diagonal_min, diagonal
