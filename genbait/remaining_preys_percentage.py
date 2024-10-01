import pandas as pd
import numpy as np

def calculate_remaining_preys_percentage(df_norm, selected_baits, n_components):
    """
    Calculates the percentage and count of remaining preys for the selected baits.

    Args:
    - df_norm (pd.DataFrame): The preprocessed data where rows represent baits and columns represent preys.
    - selected_baits (list): List of selected baits to analyze.
    - n_components (int): The number of NMF components (not used in this function, but included for consistency).

    Returns:
    - remaining_preys_percentage (float): The percentage of remaining preys in the selected subset.
    - prey_count (int): The number of preys remaining in the selected subset.
    """
    # Subset the original data to include only rows corresponding to selected baits
    original_data = df_norm.to_numpy()  # Convert the dataframe to a NumPy array for easier manipulation
    subset_indices = list(df_norm.index.get_indexer(selected_baits))  # Get the indices of selected baits
    subset_data = original_data[subset_indices, :]  # Subset the original data based on selected baits

    # Calculate the number of preys that remain (non-zero values across any baits in the subset)
    prey_count = np.count_nonzero(np.count_nonzero(subset_data, axis=0))  # Count non-zero prey across columns

    # Calculate the percentage of remaining preys
    remaining_preys_percentage = prey_count / df_norm.shape[1]  # Divide by total number of preys

    # Return the calculated percentage and prey count
    return remaining_preys_percentage, prey_count
