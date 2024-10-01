import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from scipy.optimize import linear_sum_assignment
from gprofiler import GProfiler
from concurrent.futures import ThreadPoolExecutor, as_completed

def calculate_jaccard_index(set1, set2):
    """
    Calculates the Jaccard index between two sets.

    Args:
    - set1 (set): The first set of terms.
    - set2 (set): The second set of terms.

    Returns:
    - float: The Jaccard index, which is the ratio of the intersection to the union of the two sets.
    """
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union != 0 else 0

def go_analysis_single_query(prey_names):
    """
    Performs GO analysis for a given set of prey names using the GProfiler API.

    Args:
    - prey_names (list): List of prey gene names.

    Returns:
    - set: Set of GO:CC terms for the given prey names.
    """
    gp = GProfiler(return_dataframe=True)
    try:
        go_df = gp.profile(organism='hsapiens', query=prey_names)
        go_df = go_df[go_df['source'] == 'GO:CC']
        return set(go_df['native'])
    except Exception as e:
        print(f"Error processing GO analysis for {prey_names}: {e}")
        return set()

def process_go_analysis_parallel(grouped_df):
    """
    Performs GO analysis for each cluster of preys in parallel.

    Args:
    - grouped_df (pd.core.groupby.DataFrameGroupBy): DataFrame grouped by cluster label.

    Returns:
    - dict: Dictionary where keys are cluster labels and values are sets of GO terms.
    """
    top_native_terms = {}
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_label = {executor.submit(go_analysis_single_query, list(group.index)): label for label, group in grouped_df}
        for future in as_completed(future_to_label):
            label = future_to_label[future]
            try:
                top_native_terms[label] = future.result()
            except Exception as exc:
                print(f'GO analysis generated an exception for label {label}: {exc}')
    return top_native_terms

def calculate_nmf_go_jaccard(df_norm, selected_baits, n_components, random_state=46):
    """
    Calculates the NMF GO Jaccard index for the selected baits.

    Args:
    - df_norm (pd.DataFrame): The preprocessed data.
    - selected_baits (list): List of selected baits.
    - n_components (int): The number of NMF components.
    - random_state (int): Random seed for reproducibility.

    Returns:
    - float: The mean Jaccard index across all components.
    - dict: Jaccard indices for each component.
    """
    # Subset the data for the selected baits
    subset_data = df_norm.loc[selected_baits]

    # Check if the number of selected baits is valid
    if len(selected_baits) <= n_components:
        return None, "Number of selected baits is less than or equal to the number of components."

    # Step 1: Perform NMF Clustering
    nmf = NMF(n_components=n_components, init='nndsvd', l1_ratio=1, random_state=random_state)

    # For the original dataset
    scores_matrix_original = nmf.fit_transform(df_norm)
    basis_matrix_original = nmf.components_.T

    # For the subset dataset
    scores_matrix_subset = nmf.fit_transform(subset_data)
    basis_matrix_subset = nmf.components_.T

    # Step 2: Align the components of the original and subset data using the Hungarian algorithm
    cosine_similarity = np.dot(basis_matrix_original.T, basis_matrix_subset)
    cost_matrix = 1 - cosine_similarity
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    basis_matrix_subset_reordered = basis_matrix_subset[:, col_ind]

    # Step 3: Assign labels to the original and subset components
    y_original = [np.argmax(basis_matrix_original[i, :]) for i in range(basis_matrix_original.shape[0])]
    y_subset = [np.argmax(basis_matrix_subset_reordered[i, :]) for i in range(basis_matrix_subset_reordered.shape[0])]

    # Create DataFrames for the original and subset components with labels
    basis_original_df = pd.DataFrame(basis_matrix_original, index=df_norm.columns)
    basis_subset_reordered_df = pd.DataFrame(basis_matrix_subset_reordered, index=df_norm.columns)
    basis_original_df['Label'] = y_original
    basis_subset_reordered_df['Label'] = y_subset

    # Step 4: Perform GO analysis for both the original and subset clusters
    top_native_original = process_go_analysis_parallel(basis_original_df.groupby('Label'))
    top_native_subset = process_go_analysis_parallel(basis_subset_reordered_df.groupby('Label'))

    # Step 5: Calculate the Jaccard index for each label
    jaccard_indices = {}
    for label in top_native_original:
        set1 = top_native_original[label]
        set2 = top_native_subset.get(label, set())  # Handle missing labels in the subset
        jaccard_index = calculate_jaccard_index(set1, set2)
        jaccard_indices[label] = jaccard_index

    # Step 6: Calculate the mean Jaccard index across all components
    mean_jaccard_index = np.mean(list(jaccard_indices.values()))

    # Return the mean Jaccard index and the Jaccard indices for each component
    return mean_jaccard_index, jaccard_indices
