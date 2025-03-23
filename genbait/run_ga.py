import random
import numpy as np
import pandas as pd
from deap import base, creator, tools, algorithms
from sklearn.decomposition import NMF
from scipy.optimize import linear_sum_assignment
import pickle

# Cache for previously computed fitness values to avoid redundant calculations
fitness_cache = {}

# Pre-computed values for the original data (used for NMF basis comparison)
original_data_values = {}

def precompute_original_data(df_norm, n_components):
    """
    Precompute and store NMF and other values for the original data.
    
    Args:
    - df_norm (pd.DataFrame): Normalized dataset (BioID data).
    - n_components (int): Number of NMF components.
    
    This function performs NMF decomposition on the full dataset and saves 
    the results (basis matrix and score matrix) for future use.
    """
    original_data = df_norm.to_numpy()
    nmf = NMF(n_components=n_components, init='nndsvd', l1_ratio=1, random_state=46)
    scores_matrix_original = nmf.fit_transform(original_data)  # Score matrix (samples x components)
    basis_matrix_original = nmf.components_.T  # Basis matrix (features x components)
    original_data_values['scores_matrix'] = scores_matrix_original
    original_data_values['basis_matrix'] = basis_matrix_original

def evalSubsetCorrelation(df_norm, n_components, subset_range, individual):
    """
    Evaluate the fitness of an individual (subset of features).
    
    Args:
    - df_norm (pd.DataFrame): Normalized dataset (BioID data).
    - n_components (int): Number of NMF components.
    - subset_range (tuple): Range for valid subset sizes.
    - individual (list): Binary list representing feature selection.

    Returns:
    - float: The fitness score of the individual based on correlation.
    """
    # Convert individual to a tuple to use as a dictionary key (for caching)
    individual_key = tuple(individual)
    
    # Retrieve cached fitness if available
    if individual_key in fitness_cache:
        return fitness_cache[individual_key]
    
    # Get the indices of the selected features (baits)
    subset_indices = [i for i in range(len(individual)) if individual[i] == 1]
    
    # Penalize invalid individuals if their subset size is out of the valid range
    if not (subset_range[0] <= len(subset_indices) <= subset_range[1]):
        return 0,

    # Subset the data using the selected features (baits)
    subset_data = df_norm.to_numpy()[subset_indices, :]
    nmf = NMF(n_components=n_components, init='nndsvd', l1_ratio=1, random_state=46)
    scores_matrix_subset = nmf.fit_transform(subset_data)
    basis_matrix_subset = nmf.components_.T

    # Retrieve precomputed values for the full dataset
    scores_matrix_original = original_data_values['scores_matrix']
    basis_matrix_original = original_data_values['basis_matrix']

    # Calculate cosine similarity between the basis matrices of original and subset data
    cosine_similarity = np.dot(basis_matrix_original.T, basis_matrix_subset)
    cost_matrix = 1 - cosine_similarity
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    basis_matrix_subset_reordered = basis_matrix_subset[:, col_ind]
    
    # Compute the correlation matrix between the original and reordered subset basis matrices
    corr_matrix = np.corrcoef(basis_matrix_original, basis_matrix_subset_reordered, rowvar=False)[:n_components, n_components:]
    
    # Fitness score: mean correlation of diagonal values (i.e., component-wise correlations)
    diagonal = np.diag(corr_matrix)
    diagonal_mean = np.mean(diagonal)
    penalty_factor = 0.5
    num_negative_values = np.sum(diagonal < 0)
    fitness = diagonal_mean - penalty_factor * num_negative_values,

    # Cache the computed fitness value for future use
    fitness_cache[individual_key] = fitness
    return fitness

def create_initial_individual(subset_range, n_features):
    """
    Create an individual with a random number of selected features within the subset range.
    
    Args:
    - subset_range (tuple): Range for valid subset sizes.
    - n_features (int): Total number of features in the dataset.
    
    Returns:
    - list: Binary list representing feature selection.
    """
    num_selected_features = random.randint(subset_range[0], subset_range[1])
    selected_indices = random.sample(range(n_features), num_selected_features)
    individual = [0] * n_features
    for idx in selected_indices:
        individual[idx] = 1
    return individual

def run_ga(df_norm, n_components, subset_range, population_size=500, n_generations=1000, cxpb=0.3, mutpb=0.1, seed=4):
    """
    Run the genetic algorithm to select the optimal subset of features (baits).
    
    Args:
    - df_norm (pd.DataFrame): The normalized data frame containing the BioID data.
    - n_components (int): Number of NMF components.
    - subset_range (tuple): Range for the number of selected features (min, max).
    - population_size (int): Number of individuals in the population. Default is 500.
    - n_generations (int): Number of generations to evolve. Default is 1000.
    - cxpb (float): Probability of crossover. Default is 0.3.
    - mutpb (float): Probability of mutation. Default is 0.1.
    - seed (int): Random seed for reproducibility. Default is 4.
    
    Returns:
    - pop (list): Final population after evolution.
    - logbook (deap.tools.Logbook): Logbook containing statistics of the evolution process.
    - hof (deap.tools.HallOfFame): Hall of Fame containing the best individuals.
    """
    random.seed(seed)
    np.random.seed(seed)
    precompute_original_data(df_norm, n_components)

    fitness_cache.clear()

    # Create fitness function and individual representation
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    n_features = df_norm.shape[0]

    # Lambda function to create an individual without arguments
    toolbox.register("individual", tools.initIterate, creator.Individual, 
                     lambda: create_initial_individual(subset_range, n_features))
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Register genetic operations: crossover, mutation, and selection
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("evaluate", evalSubsetCorrelation, df_norm, n_components, subset_range)

    # Initialize population, Hall of Fame (hof), and statistics tracking (logbook)
    pop = toolbox.population(n=population_size)
    hof = tools.HallOfFame(10)  # Hall of Fame stores the top 10 best individuals
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    # Run the genetic algorithm using DEAP's built-in evolutionary algorithm
    pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=cxpb, mutpb=mutpb, ngen=n_generations, stats=stats, halloffame=hof, verbose=True)

    # pop: The final population of evolved individuals.
    # logbook: Contains statistics and progress over generations (e.g., avg, min, max fitness).
    # hof: Hall of Fame storing the best-performing individuals during the evolution.

    return pop, logbook, hof

def save_ga_results(population, logbook, hof, pop_file_path='popfile.pkl', logbook_file_path='logbookfile.pkl', hof_file_path='hoffile.pkl'):
    """
    Save the results of the genetic algorithm (population and logbook) to specified file paths using pickle.
    
    Args:
    - population (list): The population resulting from the GA.
    - logbook (deap.tools.Logbook): The logbook containing statistics of the evolution process.
    - hof (deap.tools.HallOfFame): Hall of Fame containing the best individuals.
    - pop_file_path (str): Path to save the population file.
    - logbook_file_path (str): Path to save the logbook file.
    - hof_file_path (str): Path to save the Hall of Fame file.
    
    Returns:
    None
    """
    with open(pop_file_path, 'wb') as pop_file:
        pickle.dump(population, pop_file)
        
    with open(logbook_file_path, 'wb') as logbook_file:
        pickle.dump(logbook, logbook_file)

    with open(hof_file_path, 'wb') as hof_file:
        pickle.dump(hof, hof_file)

def load_ga_results(pop_file_path='popfile.pkl', logbook_file_path='logbookfile.pkl', hof_file_path='hoffile.pkl'):
    """
    Load previously saved population and logbook from their respective file paths.
    
    Args:
    - pop_file_path (str): Path to load the population file.
    - logbook_file_path (str): Path to load the logbook file.
    - hof_file_path (str): Path to load the Hall of Fame file.

    Returns:
    - population (list): The loaded population.
    - logbook (deap.tools.Logbook): The loaded logbook.
    - hof (deap.tools.HallOfFame): The loaded Hall of Fame.
    """
    with open(pop_file_path, 'rb') as pop_file:
        population = pickle.load(pop_file)
        
    with open(logbook_file_path, 'rb') as logbook_file:
        logbook = pickle.load(logbook_file)

    with open(hof_file_path, 'rb') as hof_file:
        hof = pickle.load(hof_file)
        
    return population, logbook, hof
