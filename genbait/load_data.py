import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def load_bioid_data(input_file_path, sep='\t', index_col=None, primary_baits=None, output_file_directory='data/'):
    """
    Loads and preprocesses the BioID data from a SAINT file.
    
    Args:
    - filepath (str): Path to the BioID (SAINT) data file.
    - sep (str): Delimiter for the input file (default is tab-separated).
    - index_col (int or None): Column to use as the row labels of the DataFrame.
    - primary_baits (list or None): List of primary baits to filter the dataset. If None, all baits are used.
    - file_path (str): Path to save the normalized dataset.

    Returns:
    - pd.DataFrame: Normalized BioID data after preprocessing.
    """
    # Load the raw BioID data into a DataFrame
    df = pd.read_csv(input_file_path, sep=sep, index_col=index_col)

    # Preprocess data by calculating control averages (from 'ctrlCounts')
    # ctrlCounts are split by '|' and then averaged for each bait
    ctrls = [i.split('|') for i in df['ctrlCounts']]
    sums = [sum([int(element) for element in ctrl]) / len(ctrls[0]) for ctrl in ctrls]
    df['AvgCtrl'] = sums
    
    # Subtract the average control intensity from the average spectral counts for each bait
    df['CorrectedAvgSpec'] = df['AvgSpec'] - df['AvgCtrl']
    
    # Filter rows where BFDR (Bayesian False Discovery Rate) is less than or equal to 0.01
    df = df[df['BFDR'] <= 0.01]
    
    # Reshape the data: create a pivot table where rows are baits, columns are preys, and values are corrected spectral counts
    df = df.pivot_table(index=['Bait'], columns=['PreyGene'], values=['CorrectedAvgSpec'])
    
    # Fill any missing values with 0, clip negative values to 0 (ensure no negative intensities)
    df = df.fillna(0)
    df = df.clip(lower=0)
    
    # Remove the extra column level added during the pivot operation
    df.columns = df.columns.droplevel()

    # Normalize data using MinMaxScaler (scale each prey's intensity values between 0 and 1)
    scaler = MinMaxScaler()
    df_norm = pd.DataFrame(scaler.fit_transform(df), index=df.index, columns=df.columns)

    # If primary baits are specified, filter the DataFrame to only include those baits
    if primary_baits:
        df_norm = df_norm.loc[primary_baits]
        # Further filter to exclude any prey columns that are entirely zeros across all selected baits
        df_norm = df_norm.loc[:, (df_norm != 0).any(axis=0)]

    # Save the normalized DataFrame as a CSV file
    df_norm.to_csv(f'{output_file_directory}df_norm.csv')
    
    return df_norm
