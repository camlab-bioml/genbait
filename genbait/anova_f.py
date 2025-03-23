from sklearn.feature_selection import SelectKBest, f_classif
import numpy as np
from .utils import prepare_feature_selection_data, standardize_length

def run_f_classif(df_norm, k=None, seed=4):
    """
    Perform feature selection using ANOVA F and return the top-k selected baits.

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

    selector = SelectKBest(score_func=f_classif, k=k)
    selector.fit(X_train, y_train)
    scores = selector.scores_
    selected_idx = np.argsort(scores)[-k:][::-1]
    baits = standardize_length([df_80_20.index[i] for i in selected_idx], k)

    return baits
