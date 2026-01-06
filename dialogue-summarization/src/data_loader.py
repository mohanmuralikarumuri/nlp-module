"""
Data Loader Module for Dialogue Summarization

This module handles loading and validating CSV data for dialogue summarization.
It provides utilities to:
- Load training and test data from CSV files
- Validate data integrity (required columns, missing values)
- Clean and prepare pandas DataFrames

Note: Dataset creation and tokenization are handled separately in the training step.
This module is intentionally cloud-safe and does not require PyTorch installation.
"""

import pandas as pd


def load_train_data(data_path: str) -> pd.DataFrame:
    """
    Load training dataset from CSV file with validation.
    
    This function loads dialogue-summary pairs from a CSV file and performs
    validation to ensure data integrity. It expects columns: id, dialogue, summary.
    Missing or empty values are handled safely.
    
    Args:
        data_path (str): Path to the training CSV file
        
    Returns:
        pd.DataFrame: Clean pandas DataFrame with validated data
        
    Raises:
        FileNotFoundError: If the data file doesn't exist
        ValueError: If required columns are missing
    """
    # Load CSV file
    df = pd.read_csv(data_path)
    
    # Validate required columns
    required_columns = ['id', 'dialogue', 'summary']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Handle missing values - drop rows where dialogue or summary is NaN
    initial_shape = df.shape
    df = df.dropna(subset=['dialogue', 'summary'])
    
    # Handle empty strings - filter out rows with empty dialogue or summary
    df = df[df['dialogue'].astype(str).str.strip() != '']
    df = df[df['summary'].astype(str).str.strip() != '']
    
    # Reset index after filtering
    df = df.reset_index(drop=True)
    
    # Print verification information
    print(f"Training data loaded from: {data_path}")
    print(f"Initial shape: {initial_shape}")
    print(f"After cleaning: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Sample data:")
    print(f"  - ID: {df['id'].iloc[0] if len(df) > 0 else 'N/A'}")
    print(f"  - Dialogue preview: {str(df['dialogue'].iloc[0])[:50] if len(df) > 0 else 'N/A'}...")
    print(f"  - Summary preview: {str(df['summary'].iloc[0])[:50] if len(df) > 0 else 'N/A'}...")
    
    return df


def load_test_data(data_path: str) -> pd.DataFrame:
    """
    Load test dataset from CSV file with validation.
    
    This function loads dialogue data from a CSV file. It expects columns: id, dialogue.
    The summary column is optional for test data. Missing or empty values are handled safely.
    
    Args:
        data_path (str): Path to the test CSV file
        
    Returns:
        pd.DataFrame: Clean pandas DataFrame with validated data
        
    Raises:
        FileNotFoundError: If the data file doesn't exist
        ValueError: If required columns are missing
    """
    # Load CSV file
    df = pd.read_csv(data_path)
    
    # Validate required columns (summary is optional for test data)
    required_columns = ['id', 'dialogue']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Handle missing values - drop rows where dialogue is NaN
    initial_shape = df.shape
    df = df.dropna(subset=['dialogue'])
    
    # Handle empty strings - filter out rows with empty dialogue
    df = df[df['dialogue'].astype(str).str.strip() != '']
    
    # If summary column exists, handle it gracefully (but don't require it)
    if 'summary' in df.columns:
        # For test data, summary might be intentionally empty, so just clean NaN
        df['summary'] = df['summary'].fillna('')
    
    # Reset index after filtering
    df = df.reset_index(drop=True)
    
    # Print verification information
    print(f"Test data loaded from: {data_path}")
    print(f"Initial shape: {initial_shape}")
    print(f"After cleaning: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    has_summary = 'summary' in df.columns
    print(f"Has summary column: {has_summary}")
    print(f"Sample data:")
    print(f"  - ID: {df['id'].iloc[0] if len(df) > 0 else 'N/A'}")
    print(f"  - Dialogue preview: {str(df['dialogue'].iloc[0])[:50] if len(df) > 0 else 'N/A'}...")
    if has_summary and len(df) > 0 and df['summary'].iloc[0]:
        print(f"  - Summary preview: {str(df['summary'].iloc[0])[:50]}...")
    
    return df


if __name__ == "__main__":
    """
    Main block for testing and verification of data loading functionality.
    Run with: python src/data_loader.py
    """
    import os
    
    # Define data paths relative to src directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    train_path = os.path.join(base_dir, '..', 'data', 'samsum_train.csv')
    test_path = os.path.join(base_dir, '..', 'data', 'samsum_test.csv')
    
    print("=" * 70)
    print("DATA LOADER MODULE VERIFICATION")
    print("=" * 70)
    print()
    
    # Test loading training data
    print("1. Loading Training Data...")
    print("-" * 70)
    try:
        train_df = load_train_data(train_path)
        print(f"✓ Successfully loaded training data with {len(train_df)} samples")
        print()
    except Exception as e:
        print(f"✗ Error loading training data: {e}")
        print()
    
    # Test loading test data
    print("2. Loading Test Data...")
    print("-" * 70)
    try:
        test_df = load_test_data(test_path)
        print(f"✓ Successfully loaded test data with {len(test_df)} samples")
        print()
    except Exception as e:
        print(f"✗ Error loading test data: {e}")
        print()
    
    print("=" * 70)
    print("VERIFICATION COMPLETE")
    print("=" * 70)

