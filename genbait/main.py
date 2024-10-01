import genbait as gb

# Set parameters for the analysis
datasets_path = 'data'  # Path to the dataset files
primary_baits = ['Gene1', 'Gene2', 'Gene3']  # List of primary baits to focus on
n_components = 20  # Number of components for NMF (Non-negative Matrix Factorization)
n_baits_to_select = 50  # Number of baits to be selected by the genetic algorithm

# Calculate the subset range based on n_baits_to_select
# This sets a range for how many baits to select, typically within 10 less than n_baits_to_select
subset_range = gb.calculate_subset_range(n_baits_to_select)

# Load and preprocess data from the BioID dataset
# Normalizes the data and adjusts it based on control counts
df_norm = gb.load_bioid_data(
    filepath=f'{datasets_path}/saint-latest.txt',
    primary_baits=primary_baits,
    file_path=datasets_path  # Path to save the normalized data
)

# Run Genetic Algorithm (GA) to select the optimal subset of baits
# The GA tries to find a combination of baits that maximize the correlation with the full dataset
pop, logbook, hof = gb.run_ga(
    df_norm=df_norm,  # Normalized BioID data
    n_components=n_components,  # Number of NMF components to use
    subset_range=subset_range,  # Range for the number of selected baits
)

# Save GA results (population, logbook, and hall of fame)
# The population stores all individuals, logbook stores the GA progression, and hof contains the best individuals
gb.save_ga_results(
    population=pop,
    logbook=logbook,
    hof=hof,
    pop_file_path=f"{datasets_path}/popfile.pkl",  # Path to save the population
    logbook_file_path=f"{datasets_path}/logbookfile.pkl",  # Path to save the logbook
    hof_file_path=f"{datasets_path}/hoffile.pkl"  # Path to save the hall of fame (best individuals)
)

# Load GA results (if needed for later analysis)
# population, logbook, hof = gb.load_ga_results(
#     pop_file_path=f"{datasets_path}/popfile.pkl",
#     logbook_file_path=f"{datasets_path}/logbookfile.pkl",
#     hof_file_path=f"{datasets_path}/hoffile.pkl"
# )

# Calculate GA results: best individual (optimal bait subset), mean correlation, and per-component correlations
best_individual, mean_correlation, correlations = gb.calculate_ga_results(hof, df_norm, n_components)
print(f"Best Individual (Selected Baits): {best_individual}")  # List of selected baits
print(f"Mean Correlation: {mean_correlation}")  # Mean correlation between NMF components of original and subset data
print(f"Correlations for each component: {correlations}")  # Correlation values for each component

# Calculate NMF cosine similarity between the full dataset and the selected bait subset
# Cosine similarity measures how close the NMF basis components are between the original and subset data
cosine_similarity, cosine_values = gb.calculate_nmf_cosine_similarity(df_norm, best_individual, n_components)
print(f"Mean NMF Cosine Similarity: {cosine_similarity}")  # Mean cosine similarity
print(f"Cosine Similarity values for each component: {cosine_values}")  # Cosine similarity for each component

# Calculate Adjusted Rand Index (ARI) to measure the similarity between cluster assignments
# ARI compares the clustering of the original dataset and the selected bait subset after applying NMF
ari_score = gb.calculate_nmf_ari(df_norm, best_individual, n_components)
print(f"NMF Adjusted Rand Index (ARI): {ari_score}")  # ARI score between original and subset clusters

# Calculate the KL divergence between the NMF basis components of the original and subset data
# KL divergence measures the difference in probability distributions (the closer to 0, the more similar)
mean_kl_divergence, kl_values = gb.calculate_nmf_kl_divergence(df_norm, best_individual, n_components)
print(f"Mean NMF KL Divergence: {mean_kl_divergence}")  # Mean KL divergence
print(f"KL Divergence values for each component: {kl_values}")  # KL divergence for each component

# Calculate the GO Jaccard index to measure the overlap of Gene Ontology terms between clusters
# Jaccard index measures the similarity of the GO terms of preys between the original and subset data
mean_jaccard_index = gb.calculate_nmf_go_jaccard(df_norm, best_individual, n_components)
print(f"Mean GO Jaccard Index: {mean_jaccard_index}")  # Mean Jaccard index across all components

# Calculate the percentage of remaining preys (proteins) in the selected bait subset
# This metric shows how many preys from the original dataset are retained after bait selection
remaining_percentage, remaining_count = gb.calculate_remaining_preys_percentage(df_norm, best_individual, n_components)
print(f"Remaining Preys Percentage: {remaining_percentage}")  # Percentage of remaining preys
print(f"Remaining Preys Count: {remaining_count}")  # Number of remaining preys

# Calculate the GO term retrieval percentage for the selected bait subset
# This metric evaluates how well the selected baits capture the GO terms compared to the full dataset
go_retrieval_percentage = gb.calculate_go_retrieval_percentage(df_norm, best_individual, gaf_path=f'{datasets_path}/goa_human.gaf')
print(f"GO Retrieval Percentage: {go_retrieval_percentage}")  # GO term retrieval percentage

# Evaluate clustering performance using Leiden clustering algorithm and ARI
# Leiden clustering groups data points into clusters, and ARI measures the similarity between original and subset clusters
leiden_results = gb.calculate_leiden_ari(df_norm, best_individual, resolutions=[0.5, 1.0, 1.5])
for resolution, ari in leiden_results.items():
    print(f"Resolution: {resolution}, ARI: {ari}")  # ARI for each resolution parameter

# Calculate GMM ARI for different cluster numbers
# GMM (Gaussian Mixture Model) assigns preys to clusters, and ARI compares clustering between the original and subset data
gmm_ari_results = gb.calculate_gmm_ari(df_norm, best_individual, cluster_numbers=[15, 20, 25, 30])
for cluster_number, ari in gmm_ari_results.items():
    print(f"Cluster Number: {cluster_number}, ARI: {ari}")  # ARI for different GMM cluster numbers

# Calculate GMM mean correlation between the original and subset data using soft clustering
# This metric evaluates how well the cluster probabilities match between the original and subset data
gmm_corr_results = gb.calculate_gmm_mean_correlation(df_norm, best_individual, cluster_numbers=[15, 20, 25, 30])
for cluster_number, mean_corr in gmm_corr_results.items():
    print(f"Cluster Number: {cluster_number}, Mean Correlation: {mean_corr}")  # Mean correlation for GMM clusters
