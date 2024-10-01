import pandas as pd
import os

def load_and_process_gaf(file_path):
    """
    Load and process a Gene Annotation File (GAF) to extract Gene Ontology (GO) terms.

    Args:
    - file_path (str): Path to the GAF file.

    Returns:
    - df (pd.DataFrame): A DataFrame containing the GAF data with standard column names.
    - merged_df (pd.DataFrame): A DataFrame that includes GO term sizes and associated genes.
    """
    # Load the GAF file into a DataFrame, ignoring comment lines (starting with '!').
    df = pd.read_csv(file_path, sep='\t', comment='!', header=None, dtype=str)
    
    # Set the column names based on the GAF 2.1 specification
    column_names = [
        "DB", "DB_Object_ID", "DB_Object_Symbol", "Qualifier", "GO_ID",
        "DB_Reference", "Evidence_Code", "With_or_From", "Aspect",
        "DB_Object_Name", "DB_Object_Synonym", "DB_Object_Type",
        "Taxon", "Date", "Assigned_By", "Annotation_Extension",
        "Gene_Product_Form_ID"
    ]
    df.columns = column_names[:len(df.columns)]  # Handles cases where some optional columns might be missing

    # Group by GO_ID to count the number of unique genes associated with each term (term size).
    term_size = df.groupby('GO_ID')['DB_Object_Symbol'].nunique()
    term_size = term_size.reset_index()
    term_size.columns = ['GO_ID', 'Term_Size']  # Renaming the columns for clarity
    
    # Get the associated genes for each GO term.
    associated_genes = df.groupby('GO_ID')['DB_Object_Symbol'].unique()
    associated_genes = associated_genes.reset_index()
    associated_genes.columns = ['GO_ID', 'Associated_Genes']  # Renaming columns for clarity

    # Merge the term size and associated genes data into one DataFrame.
    merged_df = pd.merge(term_size, associated_genes, on='GO_ID')
    
    return df, merged_df

def get_go_cc_for_genes(df, df_norm, genes, merged_df, max_term_size=10000):
    """
    Retrieve Gene Ontology (GO) Cellular Component (CC) terms for the selected genes, filtered by term size.

    Args:
    - df (pd.DataFrame): GAF data containing GO annotations.
    - df_norm (pd.DataFrame): Preprocessed BioID data with normalized intensities.
    - genes (list): List of genes (bait or prey) for which GO terms are needed.
    - merged_df (pd.DataFrame): DataFrame containing GO term sizes and associated genes.
    - max_term_size (int): Maximum allowed size of GO terms to be considered.

    Returns:
    - filtered_terms (list): List of GO CC terms for the given genes, filtered by term size.
    """
    # Subset the BioID data to get only the rows corresponding to the given genes.
    df_subset = df_norm.loc[genes]

    # Identify columns (preys) that have non-zero values across the selected baits.
    mask = (df_subset != 0).any(axis=0)
    df_subset_reduced = df_subset.loc[:, mask]

    # Filter the GAF data to only include GO annotations for the preys in df_subset_reduced.
    subset_df = df[df['DB_Object_Symbol'].isin(df_subset_reduced.columns)]

    # Filter the GAF data to only include annotations related to Cellular Components (CC).
    cc_df = subset_df[subset_df['Aspect'] == 'C']

    # Get the unique GO CC terms for the given genes.
    unique_go_cc_terms = cc_df['GO_ID'].unique()

    # Filter GO terms based on the maximum allowed term size (to remove overly generic terms).
    large_terms = merged_df[merged_df['Term_Size'] <= max_term_size]['GO_ID'].tolist()

    # Filter out GO CC terms that exceed the maximum term size.
    filtered_terms = [term for term in unique_go_cc_terms if term in large_terms]

    return filtered_terms

def calculate_go_retrieval_percentage(df_norm, selected_baits, gaf_path, max_term_size=10000):
    """
    Calculate the percentage of GO Cellular Component (CC) terms retrieved for the selected baits.

    Args:
    - df_norm (pd.DataFrame): The preprocessed BioID data.
    - selected_baits (list): List of selected baits.
    - gaf_path (str): Path to the Gene Annotation File (GAF) for GO terms.
    - max_term_size (int): Maximum allowed size for GO terms to avoid overly generic annotations.

    Returns:
    - overlap_subset_original (float): Percentage overlap of GO terms between the original and subset.
    """
    # Load and process the GAF file.
    df, merged_df = load_and_process_gaf(gaf_path)

    # Retrieve GO terms for the primary (original) baits.
    primary_baits = list(df_norm.index)
    go_terms_original = get_go_cc_for_genes(df, df_norm, primary_baits, merged_df, max_term_size)
    
    # Retrieve GO terms for the selected baits (subset).
    go_terms_subset = get_go_cc_for_genes(df, df_norm, selected_baits, merged_df, max_term_size)

    # Calculate the percentage overlap of GO terms between the original and the subset.
    overlap_subset_original = len(set(go_terms_subset) & set(go_terms_original)) / len(go_terms_original) * 100

    return overlap_subset_original
