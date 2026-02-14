import pandas as pd
import os

def load_data(data_path, cols=None, nrows=None):
    """
    Load Lending Club data.
    Args:
        data_path (str): Path to the CSV file.
        cols (list): List of columns to load.
        nrows (int): Number of rows to load (for testing).
    Returns:
        pd.DataFrame: Loaded dataframe.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"File not found: {data_path}")
    
    # Check text encoding
    df = pd.read_csv(data_path, usecols=cols, nrows=nrows, low_memory=False)
    
    # Random sample if nrows is specified to get representative data (simple approximation: reading first N rows)
    # A better approach for huge CSVs without reading all is not trivial without skip_rows, 
    # but for "responsiveness" first N rows is fine for dev. 
    # If we want random sampling from disk, we'd need to count lines first. 
    # For now, let's just stick to nrows for speed.
    return df

def split_data_by_time(df, date_col='issue_d'):
    """
    Split data into Train (2014-2016) and Shift (2018-2019).
    """
    # Convert date_col to datetime if not already
    df[date_col] = pd.to_datetime(df[date_col], format='%b-%Y')
    
    train_mask = (df[date_col].dt.year >= 2014) & (df[date_col].dt.year <= 2016)
    shift_mask = (df[date_col].dt.year >= 2018) & (df[date_col].dt.year <= 2019)
    
    train_df = df[train_mask].copy()
    shift_df = df[shift_mask].copy()
    
    return train_df, shift_df
