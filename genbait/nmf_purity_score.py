import pandas as pd
import numpy as np
from sklearn.decomposition import NMF
from scipy.optimize import linear_sum_assignment

def calculate_min_preservation(df_norm, df_subset, n_components, random_state=46):
    """
    Calculates the minimum preservation of preys for each NMF component
    between the original and subset datasets.

    Args:
    - df_norm (pd.DataFrame): The original dataset.
    - df_subset (pd.DataFrame): The subset of the dataset based on selected baits.
    - n_components (int): Number of NMF components.
    - random_state (int): Random seed for reproducibility.

    Returns:
    - float: The minimum component-wise preservation score.
    """
    # Filter zero-only columns in subset
    mask = (df_subset != 0).any(axis=0)
    df_subset_reduced = df_subset.loc[:, mask]

    # Run NMF on original dataset
    nmf = NMF(n_components=n_components, init='nndsvd', l1_ratio=1, random_state=random_state)
    scores_matrix_original = nmf.fit_transform(df_norm)
    basis_matrix_original = nmf.components_.T

    # Run NMF on subset dataset
    scores_matrix_subset = nmf.fit_transform(df_subset)
    basis_matrix_subset = nmf.components_.T

    # Align components using Hungarian algorithm
    cosine_similarity = np.dot(basis_matrix_original.T, basis_matrix_subset)
    cost_matrix = 1 - cosine_similarity
    _, col_ind = linear_sum_assignment(cost_matrix)
    basis_matrix_subset_reordered = basis_matrix_subset[:, col_ind]

    # Convert to DataFrames with prey names as index
    basis_original_df = pd.DataFrame(basis_matrix_original, index=df_norm.columns)
    basis_subset_df = pd.DataFrame(basis_matrix_subset_reordered, index=df_subset.columns)

    # Intersect common preys
    common_index = basis_original_df.index.intersection(basis_subset_df.index)
    common_original = basis_original_df.loc[common_index].to_numpy()
    common_subset = basis_subset_df.loc[common_index].to_numpy()

    # Compute preservation for each component
    min_preservation = 1.0
    for k in range(n_components):
        original_preys = np.where(np.argmax(common_original, axis=1) == k)[0]
        if len(original_preys) == 0:
            continue  # skip empty component
        subset_preys = np.where(np.argmax(common_subset, axis=1) == k)[0]
        preserved = len(set(original_preys).intersection(subset_preys))
        preservation = preserved / len(original_preys)
        if preservation < min_preservation:
            min_preservation = preservation

    return min_preservation

def calculate_min_nmf_purity(df_norm, selected_baits, n_components):
    """
    Calculates the minimum NMF component purity (prey preservation) for selected baits.

    Args:
    - df_norm (pd.DataFrame): The preprocessed data.
    - selected_baits (list): List of selected baits.
    - n_components (int): The number of NMF components.

    Returns:
    - float: Minimum prey preservation across NMF components.
    """
    # Subset using selected baits
    subset_data = df_norm.loc[selected_baits]

    # Check if enough baits for components
    if len(selected_baits) <= n_components:
        return None, "Number of selected baits is less than or equal to the number of components."

    # Compute minimum preservation score
    min_purity = calculate_min_preservation(df_norm, subset_data, n_components)

    return min_purity
