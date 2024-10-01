import pandas as pd
import numpy as np
import random
from sklearn.metrics import adjusted_rand_score
from sklearn.neighbors import kneighbors_graph
import igraph as ig
import leidenalg

def create_knn_graph(data, k=20):
    """
    Create a k-nearest neighbors (kNN) graph using the input data.

    Args:
    - data (numpy.ndarray): The input data matrix where each row is a sample.
    - k (int): The number of neighbors for the kNN graph (default is 20).

    Returns:
    - igraph.Graph: The constructed kNN graph with edge weights.
    """
    # Create a kNN graph from the input data using sklearn's kneighbors_graph
    knn_graph = kneighbors_graph(data, k, mode='connectivity', include_self=False).toarray() # type: ignore
    
    # Extract the source and target indices of the graph edges
    sources, targets = knn_graph.nonzero()
    
    # Get the edge weights
    weights = knn_graph[sources, targets]
    
    # Initialize an igraph graph with vertices corresponding to the samples
    g = ig.Graph(directed=False)
    g.add_vertices(data.shape[0])
    
    # Add edges to the graph
    edges = list(zip(sources, targets))
    g.add_edges(edges)
    
    # Assign the weights to the edges
    g.es['weight'] = weights
    
    return g

def leiden_clustering(graph, resolution):
    """
    Perform Leiden clustering on the input graph.

    Args:
    - graph (igraph.Graph): The graph to perform clustering on.
    - resolution (float): The resolution parameter for the Leiden algorithm, controlling the granularity of the clustering.

    Returns:
    - list: The membership (cluster assignments) of each vertex in the graph.
    """
    # Use the RBConfigurationVertexPartition method to perform Leiden clustering with the provided resolution
    partition = leidenalg.find_partition(graph, leidenalg.RBConfigurationVertexPartition, 
                                         weights='weight', resolution_parameter=resolution)
    
    return partition.membership

def precalculate_original_clusters(df_original, resolutions):
    """
    Precompute the Leiden clusters for the original dataset at multiple resolutions.

    Args:
    - df_original (pd.DataFrame): The preprocessed data for the original dataset.
    - resolutions (list): List of resolutions to compute the clustering at.

    Returns:
    - dict: A dictionary where each key is a resolution, and the value is the cluster assignments for that resolution.
    """
    # Transpose the data so that rows correspond to samples
    df_original_transposed = df_original.transpose()
    
    # Dictionary to store clusters for different resolutions
    original_clusters_resolutions = {}
    
    # Loop through each resolution
    for resolution in resolutions:
        # Create a kNN graph from the original data
        original_graph = create_knn_graph(df_original_transposed, k=20)
        
        # Perform Leiden clustering on the graph
        original_clusters = leiden_clustering(original_graph, resolution)
        
        # Store the clusters for this resolution
        original_clusters_resolutions[resolution] = original_clusters
    
    return original_clusters_resolutions

def calculate_ari_for_subset(df_original, df_subset, original_clusters_resolutions, resolution):
    """
    Calculate the Adjusted Rand Index (ARI) between clusters of the original data and a subset.

    Args:
    - df_original (pd.DataFrame): The preprocessed data for the original dataset.
    - df_subset (pd.DataFrame): The subset of the data.
    - original_clusters_resolutions (dict): The precomputed cluster assignments for the original data.
    - resolution (float): The resolution at which clustering was performed.

    Returns:
    - float: The ARI score between the original and subset cluster assignments.
    """
    # Transpose the subset data
    df_subset_transposed = df_subset.transpose()
    
    # Create a kNN graph for the subset
    subset_graph = create_knn_graph(df_subset_transposed, k=20)
    
    # Perform Leiden clustering on the subset graph
    subset_clusters = leiden_clustering(subset_graph, resolution)
    
    # Find the common samples between the original and the subset data
    common_samples = df_original.columns.intersection(df_subset.columns)
    
    # Get the cluster assignments for the original and subset data for the common samples
    original_clusters = original_clusters_resolutions[resolution]
    original_index_map = {sample: index for index, sample in enumerate(df_original.columns)}
    subset_index_map = {sample: index for index, sample in enumerate(df_subset.columns)}
    
    # Extract the cluster assignments for the common samples
    common_original_clusters = [original_clusters[original_index_map[sample]] for sample in common_samples]
    common_subset_clusters = [subset_clusters[subset_index_map[sample]] for sample in common_samples]
    
    # Calculate and return the ARI score
    return adjusted_rand_score(common_original_clusters, common_subset_clusters)

def calculate_leiden_ari(df_norm, selected_baits, resolutions, seed=4):
    """
    Calculate the Adjusted Rand Index (ARI) for Leiden clustering between the original data and the selected baits.

    Args:
    - df_norm (pd.DataFrame): The preprocessed data.
    - selected_baits (list): List of selected baits to analyze.
    - resolutions (list): List of resolutions for Leiden clustering.
    - seed (int): Random seed for reproducibility (default is 4).

    Returns:
    - dict: A dictionary with resolution as keys and ARI values as values.
    """
    # Set random seed for reproducibility
    random.seed(seed)
    
    # Subset the data to include only the selected baits
    df_subset = df_norm.loc[selected_baits]
    
    # Mask to remove columns (preys) that have all zero values
    mask = (df_subset != 0).any(axis=0)
    df_subset = df_subset.loc[:, mask]
    
    # Precompute clusters for the original dataset for all resolutions
    original_clusters_resolutions = precalculate_original_clusters(df_norm, resolutions)
    
    # Dictionary to store the ARI results for each resolution
    results = {}
    
    # For each resolution, calculate the ARI between the original data and the subset
    for resolution in resolutions:
        ari = calculate_ari_for_subset(df_norm, df_subset, original_clusters_resolutions, resolution)
        results[resolution] = ari
    
    return results
