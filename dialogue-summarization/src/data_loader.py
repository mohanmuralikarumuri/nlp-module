"""
Data Loader Module for Dialogue Summarization

This module handles loading and batching of dialogue-summary datasets.
It provides utilities to:
- Load training and test data from CSV files
- Create PyTorch DataLoaders for efficient batching
- Handle data transformations and tokenization
"""

from typing import Optional, Tuple
import pandas as pd

try:
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    # Define dummy classes if torch is not available
    class Dataset:
        """Dummy Dataset class when torch is not available."""
        pass
    
    class DataLoader:
        """Dummy DataLoader class when torch is not available."""
        pass


class DialogueSummarizationDataset(Dataset):
    """
    Custom Dataset class for dialogue summarization task.
    
    Handles loading dialogues and summaries from CSV format and
    preparing them for model training.
    """
    
    def __init__(self, data_path, tokenizer=None):
        """
        Initialize the dataset.
        
        Args:
            data_path (str): Path to the CSV file containing dialogues
            tokenizer: Tokenizer instance for text encoding
        """
        self.data = pd.read_csv(data_path)
        self.tokenizer = tokenizer
    
    def __len__(self):
        """Return the total number of samples in the dataset."""
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Get a single sample from the dataset.
        
        Args:
            idx (int): Index of the sample to retrieve
            
        Returns:
            dict: Dictionary containing dialogue and summary (if available)
        """
        # Implementation to be added
        pass


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
    df = df[df['dialogue'].str.strip().astype(bool)]
    df = df[df['summary'].str.strip().astype(bool)]
    
    # Reset index after filtering
    df = df.reset_index(drop=True)
    
    # Print verification information
    print(f"Training data loaded from: {data_path}")
    print(f"Initial shape: {initial_shape}")
    print(f"After cleaning: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Sample data:")
    print(f"  - ID: {df['id'].iloc[0] if len(df) > 0 else 'N/A'}")
    print(f"  - Dialogue preview: {df['dialogue'].iloc[0][:50] if len(df) > 0 else 'N/A'}...")
    print(f"  - Summary preview: {df['summary'].iloc[0][:50] if len(df) > 0 else 'N/A'}...")
    
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
    df = df[df['dialogue'].str.strip().astype(bool)]
    
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
    print(f"  - Dialogue preview: {df['dialogue'].iloc[0][:50] if len(df) > 0 else 'N/A'}...")
    if has_summary and len(df) > 0:
        print(f"  - Summary preview: {df['summary'].iloc[0][:50] if df['summary'].iloc[0] else 'N/A'}...")
    
    return df


def create_data_loader(data_path, tokenizer, batch_size=8, shuffle=True):
    """
    Create a DataLoader for dialogue summarization dataset.
    
    Args:
        data_path (str): Path to the CSV data file
        tokenizer: Tokenizer instance for text encoding
        batch_size (int): Number of samples per batch
        shuffle (bool): Whether to shuffle the data
        
    Returns:
        DataLoader: PyTorch DataLoader instance
    """
    dataset = DialogueSummarizationDataset(data_path, tokenizer)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


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

