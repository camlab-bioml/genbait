import os
import pandas as pd
import numpy as np
from sklearn.decomposition import NMF
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.pairwise import cosine_similarity

def calculate_nmf_cosine_similarity(df_norm, selected_baits, n_components):
    """
    Calculates the NMF mean cosine similarity for the selected baits.

    Args:
    - df_norm (pd.DataFrame): The preprocessed data.
    - selected_baits (list): List of selected baits.
    - n_components (int): The number of NMF components.

    Returns:
    - diagonal_mean (float): Mean cosine similarity between the original and subset NMF components.
    - diagonal (np.ndarray): Array of cosine similarity values for each component.
    """
    
    # Subset the data using the selected baits
    original_data = df_norm.to_numpy()
    
    # Get indices of the selected baits
    subset_indices = list(df_norm.index.get_indexer(selected_baits))
    subset_data = original_data[subset_indices, :]

    # Check if the number of selected baits is valid (should be greater than n_components)
    if len(subset_indices) <= n_components:
        return None, "Number of selected baits is less than or equal to the number of components."

    # Apply NMF to the original data
    nmf = NMF(n_components=n_components, init='nndsvd', l1_ratio=1, random_state=46)
    scores_matrix_original = nmf.fit_transform(original_data)
    basis_matrix_original = nmf.components_.T

    # Apply NMF to the subset data (selected baits)
    scores_matrix_subset = nmf.fit_transform(subset_data)
    basis_matrix_subset = nmf.components_.T

    # Calculate cosine similarity matrix and apply the Hungarian algorithm
    cosine_similarity_matrix = np.dot(basis_matrix_original.T, basis_matrix_subset)
    cost_matrix = 1 - cosine_similarity_matrix
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    basis_matrix_subset_reordered = basis_matrix_subset[:, col_ind]

    # Compute the cosine similarity between the original and reordered subset components
    cos_sim_matrix = cosine_similarity(basis_matrix_original.T, basis_matrix_subset_reordered.T)

    # Extract the diagonal values (cosine similarity for each component)
    diagonal = np.diag(cos_sim_matrix)
    
    # Calculate the mean cosine similarity of the diagonal values
    diagonal_mean = np.mean(diagonal)

    # Calculate the min cosine similarity of the diagonal values
    diagonal_min = np.min(diagonal)

    # Return the mean cosine similarity and the cosine similarity values for each component
    return diagonal_mean, diagonal_min, diagonal
