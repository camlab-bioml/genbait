import pandas as pd
import numpy as np
from sklearn.decomposition import NMF
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score

def calculate_ari(df_norm, df_subset, n_components, random_state=46):
    """
    Calculates the Adjusted Rand Index (ARI) between NMF clusters of the original and subset data.

    Args:
    - df_norm (pd.DataFrame): The original dataset.
    - df_subset (pd.DataFrame): The subset of the dataset based on selected baits.
    - n_components (int): Number of NMF components.
    - random_state (int): Random seed for reproducibility.

    Returns:
    - float: The ARI score between the original and subset NMF clusters.
    """
    # Filter the subset data to remove columns with all zero values
    mask = (df_subset != 0).any(axis=0)
    df_subset_reduced = df_subset.loc[:, mask]

    # Perform NMF on the original dataset
    nmf = NMF(n_components=n_components, init='nndsvd', l1_ratio=1, random_state=random_state)
    scores_matrix_original = nmf.fit_transform(df_norm)
    basis_matrix_original = nmf.components_.T

    # Perform NMF on the subset dataset
    scores_matrix_subset = nmf.fit_transform(df_subset)
    basis_matrix_subset = nmf.components_.T

    # Align the components of the original and subset data using the Hungarian algorithm
    cosine_similarity = np.dot(basis_matrix_original.T, basis_matrix_subset)
    cost_matrix = 1 - cosine_similarity
    _, col_ind = linear_sum_assignment(cost_matrix)
    basis_matrix_subset_reordered = basis_matrix_subset[:, col_ind]

    # Get the column names of df_subset_reduced
    reduced_cols = df_subset_reduced.columns

    # Reorder the subset data to match the original data alignment
    basis_matrix_subset_reordered_df = pd.DataFrame(basis_matrix_subset_reordered, 
                                                    columns=[i for i in range(basis_matrix_subset_reordered.shape[1])])

    # Filter the reordered matrix to match reduced columns
    reduced_indices = [df_subset.columns.get_loc(c) for c in reduced_cols]
    basis_matrix_subset_reordered_reduced = basis_matrix_subset_reordered_df.iloc[reduced_indices, :].to_numpy()

    # Assign labels to the original and reordered subset data based on max NMF scores
    y_original = np.argmax(basis_matrix_original, axis=1)
    y_subset = np.argmax(basis_matrix_subset_reordered_reduced, axis=1)

    # Create DataFrames for original and subset data, with assigned labels
    basis_original_df = pd.DataFrame(basis_matrix_original, index=df_norm.columns)
    basis_subset_reordered_reduced_df = pd.DataFrame(basis_matrix_subset_reordered_reduced, index=df_subset_reduced.columns)
    basis_original_df['Label'] = y_original
    basis_subset_reordered_reduced_df['Label'] = y_subset

    # Step 1: Identify common indices between original and subset data
    common_indices = basis_original_df.index.intersection(basis_subset_reordered_reduced_df.index)

    # Step 2: Extract the labels for common indices
    labels_original = basis_original_df.loc[common_indices, 'Label']
    labels_subset = basis_subset_reordered_reduced_df.loc[common_indices, 'Label']

    # Step 3: Calculate the ARI between the original and subset labels
    ari_score = adjusted_rand_score(labels_original, labels_subset)

    return ari_score

def calculate_nmf_ari(df_norm, selected_baits, n_components):
    """
    Calculates the NMF ARI (Adjusted Rand Index) for the selected baits.

    Args:
    - df_norm (pd.DataFrame): The preprocessed data.
    - selected_baits (list): List of selected baits.
    - n_components (int): The number of NMF components.

    Returns:
    - float: The ARI score between the original and subset NMF clusters.
    """
    # Subset the data using the selected baits
    subset_data = df_norm.loc[selected_baits]

    # Check if the number of selected baits is sufficient
    if len(selected_baits) <= n_components:
        return None, "Number of selected baits is less than or equal to the number of components."

    # Calculate ARI using the calculate_ari function
    ari_score = calculate_ari(df_norm, subset_data, n_components)

    # Return the ARI score
    return ari_score
