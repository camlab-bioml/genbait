import genbait as gb

# Set parameters for the analysis
input_directory = 'input_files/' # path to SAINT inout file and other inputs
output_directory = 'output_files/'  # path to save the outpus

# primary_baits = ['Gene1', 'Gene2', 'Gene3']  # List of primary baits to focus on

n_components = 20  # Number of components for NMF (Non-negative Matrix Factorization)
n_baits_to_select = 50  # Number of baits to be selected by the genetic algorithm

# Calculate the subset range based on n_baits_to_select
# This sets a range for how many baits to select, typically within 10 less than n_baits_to_select
subset_range = gb.calculate_subset_range(n_baits_to_select)


# Load and preprocess data from the BioID dataset
# Normalizes the data and adjusts it based on control counts
df_norm = gb.load_bioid_data(
    input_file_path=f'{input_directory}saint-latest.txt',
    output_file_directory=output_directory
)

# Run Genetic Algorithm (GA) to select the optimal subset of baits
# The GA tries to find a combination of baits that maximize the correlation with the full dataset
pop, logbook, hof = gb.run_ga(
    df_norm=df_norm,
    n_components=n_components,
    subset_range=subset_range,
    population_size=50,
    n_generations=3
)

# Save GA results (population, logbook, and hall of fame)
# The population stores all individuals, logbook stores the GA progression, and hof contains the best individuals
gb.save_ga_results(
    population=pop,
    logbook=logbook,
    hof=hof,
    pop_file_path=f"{output_directory}/popfile.pkl",
    logbook_file_path=f"{output_directory}/logbookfile.pkl",
    hof_file_path=f"{output_directory}/hoffile.pkl"
)

## Load GA results (if needed for later analysis)
# population, logbook, hof = gb.load_ga_results(
#     pop_file_path=f"{output_directory}/popfile.pkl",
#     logbook_file_path=f"{output_directory}/logbookfile.pkl",
#     hof_file_path=f"{output_directory}/hoffile.pkl"
# )

# Calculate GA results: best individual (optimal bait subset), mean correlation, and per-component correlations
selected_baits, mean_nmf_correlation, all_nmf_correaltions = gb.calculate_ga_results(hof, df_norm, n_components)
print(f"Best Individual (Selected Baits): {selected_baits}")  # List of selected baits
print(f"Mean Correlation: {mean_nmf_correlation}")  # Mean correlation between NMF components of original and subset data
print(f"Correlations for each component: {all_nmf_correaltions}")  # Correlation values for each component

# Calculate NMF cosine similarity between the full dataset and the selected bait subset
# Cosine similarity measures how close the NMF basis components are between the original and subset data
mean_nmf_cos_similarity_score, all_nmf_cos_similarity_scores = gb.calculate_nmf_cosine_similarity(df_norm, selected_baits, n_components)
print(f"Mean NMF Cosine Similarity: {mean_nmf_cos_similarity_score}")  # Mean cosine similarity
print(f"Cosine Similarity values for each component: {all_nmf_cos_similarity_scores}")  # Cosine similarity for each component

# Calculate the KL divergence between the NMF basis components of the original and subset data
# KL divergence measures the difference in probability distributions (the closer to 0, the more similar)
mean_nmf_kl_divergence_score, all_nmf_kl_divergence_scores = gb.calculate_nmf_kl_divergence(df_norm, selected_baits, n_components)
print(f"Mean NMF KL Divergence: {mean_nmf_kl_divergence_score}")  # Mean KL divergence
print(f"KL Divergence values for each component: {all_nmf_kl_divergence_scores}")  # KL divergence for each component

# Calculate the GO Jaccard index to measure the overlap of Gene Ontology terms between clusters
# Jaccard index measures the similarity of the GO terms of preys between the original and subset data
mean_nmf_go_jaccard_index, all_nmf_go_jaccard_indices = gb.calculate_nmf_go_jaccard(df_norm, selected_baits, n_components=n_components)
print(f"Mean NMF GO Jaccard index: {mean_nmf_go_jaccard_index}")  # Mean KL divergence
print(f"GO Jaccard index values for each component: {all_nmf_go_jaccard_indices}")  # KL divergence for each component

# Calculate Adjusted Rand Index (ARI) to measure the similarity between cluster assignments
# ARI compares the clustering of the original dataset and the selected bait subset after applying NMF
nmf_ari_score = gb.calculate_nmf_ari(df_norm, selected_baits, n_components)
print(f"NMF Adjusted Rand Index (ARI): {nmf_ari_score}")  # ARI score between original and subset clusters



# Calculate the percentage of remaining preys (proteins) in the selected bait subset
# This metric shows how many preys from the original dataset are retained after bait selection
remaining_preys_percentage, remaining_preys_count = gb.calculate_remaining_preys_percentage(df_norm, selected_baits, n_components)
print(f"Remaining Preys Percentage: {remaining_preys_percentage}")  # Percentage of remaining preys
print(f"Remaining Preys Count: {remaining_preys_count}")  # Number of remaining preys

# Calculate the GO term retrieval percentage for the selected bait subset
# This metric evaluates how well the selected baits capture the GO terms compared to the full dataset
go_retrieval_percentage = gb.calculate_go_retrieval_percentage(df_norm, selected_baits, gaf_path=f'{input_directory}goa_human.gaf')
print(f"GO Retrieval Percentage: {go_retrieval_percentage}")  # GO term retrieval percentage

# Evaluate clustering performance using Leiden clustering algorithm and ARI
# Leiden clustering groups data points into clusters, and ARI measures the similarity between original and subset clusters
leiden_results = gb.calculate_leiden_ari(df_norm, selected_baits, resolutions=[0.5, 1.0, 1.5])
for resolution, ari in leiden_results.items():
    print(f"Resolution: {resolution}, ARI: {ari}")  # ARI for each resolution parameter

# Calculate GMM ARI for different cluster numbers
# GMM (Gaussian Mixture Model) assigns preys to clusters, and ARI compares clustering between the original and subset data
gmm_ari_results = gb.calculate_gmm_ari(df_norm, selected_baits, cluster_numbers=[15, 20, 25, 30])
for cluster_number, ari in gmm_ari_results.items():
    print(f"Cluster Number: {cluster_number}, ARI: {ari}")  # ARI for different GMM cluster numbers

# Calculate GMM mean correlation between the original and subset data using soft clustering
# This metric evaluates how well the cluster probabilities match between the original and subset data
gmm_corr_results = gb.calculate_gmm_mean_correlation(df_norm, selected_baits, cluster_numbers=[15, 20, 25, 30])
for cluster_number, mean_corr in gmm_corr_results.items():
    print(f"Cluster Number: {cluster_number}, Mean Correlation: {mean_corr}")  # Mean correlation for GMM clusters
