import torch
import torch.nn as nn
import torch.optim as optim
import shap
import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from scipy.optimize import nnls
import pytorch_lightning as pl

def calculate_subset_range(n_baits_to_select, delta=10):
    """
    Calculate the valid range of subset sizes for bait selection.

    This function defines a range of subset sizes based on the number of desired baits to select.
    It provides a range from (n_baits_to_select - delta) to n_baits_to_select.

    Args:
    - n_baits_to_select (int): The target number of baits to be selected.
    - delta (int, optional): The allowable range around the target number of baits to be selected.
      Default is 10.

    Returns:
    - tuple: A tuple containing the minimum and maximum subset size (min_size, max_size).
    """
    # Returns a tuple of the subset range, i.e., (n_baits_to_select - delta, n_baits_to_select).
    # delta specifies the buffer on how much smaller the subset size can be.
    return (n_baits_to_select - delta, n_baits_to_select)





class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class LightningNN(pl.LightningModule):
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.001):
        super(LightningNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.learning_rate = learning_rate

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

    def training_step(self, batch, batch_idx):
        X_batch, y_batch = batch
        outputs = self(X_batch)
        loss = nn.CrossEntropyLoss()(outputs, y_batch)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.learning_rate)


def train_lightning_nn(X_train, y_train, input_size, hidden_size, output_size, epochs=10, batch_size=32, learning_rate=0.001):
    if isinstance(X_train, pd.DataFrame):
        X_train = X_train.to_numpy()
    if isinstance(y_train, pd.Series):
        y_train = y_train.to_numpy()

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)

    dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = LightningNN(input_size, hidden_size, output_size, learning_rate)
    trainer = pl.Trainer(max_epochs=epochs, enable_checkpointing=False, logger=False)
    trainer.fit(model, dataloader)

    return model

def get_feature_importance_shap_lightning(model, X_train):
    if isinstance(X_train, pd.DataFrame):
        X_train = X_train.to_numpy()

    model.eval()

    def model_forward(x):
        with torch.no_grad():
            return model(torch.tensor(x, dtype=torch.float32)).detach().numpy()

    explainer = shap.Explainer(model_forward, X_train)
    shap_values = explainer(X_train)

    feature_importances = np.abs(shap_values.values).mean(axis=0).sum(axis=1)

    return feature_importances

def prepare_feature_selection_data(df_norm, seed=4):
    np.random.seed(seed)

    # Splitting the data
    num_cols = df_norm.shape[1]
    num_split = int(num_cols * 0.8)
    cols = df_norm.columns.tolist()
    np.random.shuffle(cols)
    cols_80 = cols[:num_split]
    cols_20 = cols[num_split:]

    df_80 = df_norm[cols_80]
    df_20 = df_norm[cols_20]

    # NMF Decomposition for entire df_norm
    nmf = NMF(n_components=20, init='nndsvd', l1_ratio=1, random_state=46)
    scores = nmf.fit_transform(df_norm)
    basis = nmf.components_.T

    scores_df = pd.DataFrame(scores)
    basis_df = pd.DataFrame(basis)

    X = df_norm.T.to_numpy()
    y = []
    for i in range(basis.shape[0]):
        max_rank = np.argmax(basis[i,:]) 
        y.append(max_rank)
    y = np.asarray(y) 

    scores_80 = nmf.fit_transform(df_80)
    basis_80 = nmf.components_.T

    scores_80_df = pd.DataFrame(scores_80)
    basis_80_df = pd.DataFrame(basis_80)

    X = df_norm.T.to_numpy()
    y_80 = []
    for i in range(basis_80.shape[0]):
        max_rank = np.argmax(basis_80[i,:]) 
        y_80.append(max_rank)
    y_80 = np.asarray(y_80)    
    # Initialize basis_20 with zeros (its dimensions will be n x p)
    basis_20 = np.zeros((scores_80.shape[1], df_20.shape[1]))

    # Iterate over each column in df_20 and solve for each column in basis_20
    for i in range(df_20.shape[1]):
        basis_20[:, i], _ = nnls(scores_80, df_20.iloc[:, i])

    # Now, basis_20 should be the matrix you're looking for

    basis_20 = basis_20.T
    basis_20_df = pd.DataFrame(basis_20)
    df_80_20 = pd.concat([df_80, df_20], axis=1)
    basis_80_20 = pd.concat([basis_80_df, basis_20_df], axis=0)
    basis_80_20.reset_index(inplace=True)

    y_20 = []
    for i in range(basis_20.shape[0]):
        max_rank = np.argmax(basis_20[i,:]) 
        y_20.append(max_rank)
    y_20 = np.asarray(y_20)   

    X_train, X_test, y_train, y_test = df_80.T, df_20.T, y_80, y_20

    return X_train, y_train, df_80_20


def standardize_length(lst, target_length, fill_value=None):
    return lst[:target_length] + [fill_value] * (target_length - len(lst))