import xgboost as xgb
import numpy as np
from .utils import prepare_feature_selection_data, standardize_length

def run_xgb(df_norm, k=None, seed=4):
    """
    Perform feature selection using XGB and return the top-k selected baits.

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

    model = xgb.XGBClassifier(use_label_encoder=False, random_state=46, eval_metric='mlogloss')
    model.fit(X_train, y_train)
    importances = model.feature_importances_
    selected_idx = np.argsort(importances)[::-1][:k]
    baits = standardize_length([df_80_20.index[i] for i in selected_idx], k)

    return baits
