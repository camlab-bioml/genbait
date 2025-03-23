import pandas as pd
import numpy as np
from sklearn.decomposition import NMF
from scipy.optimize import linear_sum_assignment

def calculate_ga_results(hof, df_norm, n_components):
    """
    Calculate the results of the genetic algorithm by performing NMF on the selected baits and 
    comparing the NMF components with those from the original dataset.

    Args:
    - hof (deap.tools.HallOfFame): The hall of fame containing the best individuals (subsets of baits).
    - df_norm (pd.DataFrame): The preprocessed normalized data containing bait-prey intensities.
    - n_components (int): The number of NMF components to use.

    Returns:
    - selected_baits (list): List of selected baits (features) chosen by the genetic algorithm.
    - diagonal_mean (float): Mean correlation between NMF components of the original and subset data.
    - diagonal_min (float): Min correlation between NMF components of the original and subset data.
    - diagonal (np.ndarray): Array of correlation values for each component.
    """

    # Extract the best individual from the hall of fame (subset of baits chosen by the genetic algorithm)
    best_individual = hof[0]
    
    # Select the baits that were included in the best individual's selection
    selected_baits = [df_norm.index[i] for i in range(len(best_individual)) if best_individual[i] == 1]

    # Get the indices of the selected baits
    subset_indices = [i for i in range(len(best_individual)) if best_individual[i] == 1]
    
    # Subset the data for the selected baits
    subset_data = df_norm.to_numpy()[subset_indices, :]

    # Perform NMF on the subset of data (selected baits)
    nmf = NMF(n_components=n_components, init='nndsvd', l1_ratio=1, random_state=46)
    scores_matrix_subset = nmf.fit_transform(subset_data)
    basis_matrix_subset = nmf.components_.T

    # Perform NMF on the entire original dataset
    original_data = df_norm.to_numpy()
    nmf_original = NMF(n_components=n_components, init='nndsvd', l1_ratio=1, random_state=46)
    scores_matrix_original = nmf_original.fit_transform(original_data)
    basis_matrix_original = nmf_original.components_.T

    # Calculate the cosine similarity between the original and subset NMF components
    cosine_similarity = np.dot(basis_matrix_original.T, basis_matrix_subset)

    # Use the Hungarian algorithm to find the optimal assignment between original and subset components
    cost_matrix = 1 - cosine_similarity
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    basis_matrix_subset_reordered = basis_matrix_subset[:, col_ind]

    # Compute the correlation matrix between the original and reordered subset components
    corr_matrix = np.corrcoef(basis_matrix_original, basis_matrix_subset_reordered, rowvar=False)[:n_components, n_components:]

    # Extract the diagonal of the correlation matrix (correlation values for each component)
    diagonal = np.diag(corr_matrix)
    
    # Calculate the mean correlation of the diagonal values
    diagonal_mean = np.mean(diagonal)

    # Calculate the min correlation of the diagonal values
    diagonal_min = np.min(diagonal)

    # Return the selected baits, the mean correlation, and the correlation values for each component
    return selected_baits, diagonal_mean, diagonal_min, diagonal

