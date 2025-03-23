import numpy as np
from .utils import prepare_feature_selection_data, train_lightning_nn, get_feature_importance_shap_lightning, standardize_length

def run_nn(df_norm, k=None, seed=4):
    """
    Perform feature selection using Neural network and return the top-k selected baits.

    Args:
    - df_norm: input dataframe (baits as rows, preys as columns).
    - k (int or None): Number of baits to select. If None, selects all baits in df_80_20.
    - seed (int): Random seed for reproducibility (used in data splitting and shuffling).

    Returns:
    - baits (list of str): List of selected bait names (row indices from df_norm).
    """
    X_train, y_train, df_80_20 = prepare_feature_selection_data(df_norm, seed)
    if k is None:
        k = len(df_80_20)

    input_size = X_train.shape[1]
    hidden_size = 64
    output_size = len(np.unique(y_train))
    
    model = train_lightning_nn(X_train, y_train, input_size, hidden_size, output_size)
    shap_scores = get_feature_importance_shap_lightning(model, X_train)
    
    selected_idx = np.argsort(shap_scores)[::-1][:k]
    baits = standardize_length([df_80_20.index[i] for i in selected_idx], k)

    return baits
