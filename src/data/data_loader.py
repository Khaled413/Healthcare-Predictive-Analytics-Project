import pandas as pd
import os

def load_data(filepath):
    """
    Load data from CSV file
    
    Args:
        filepath: Path to the CSV file
        
    Returns:
        DataFrame with loaded data
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file not found at {filepath}")
    
    # Load data
    df = pd.read_csv(filepath)
    
    # Print basic information
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    return df

def get_feature_target_split(df, target_column):
    """
    Split dataframe into features and target
    
    Args:
        df: DataFrame with data
        target_column: Name of the target column
        
    Returns:
        X, y: Features and target
    """
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataframe")
    
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    return X, y