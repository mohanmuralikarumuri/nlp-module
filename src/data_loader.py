"""
Data Loader Module

This module handles loading and preparing dialogue-summarization datasets
from CSV files with proper validation and error handling.

Expected CSV format:
    - id: Unique identifier for each dialogue
    - dialogue: Input conversation text
    - summary: Target summary (only in training data)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List
import warnings

try:
    from datasets import Dataset, DatasetDict
    HF_DATASETS_AVAILABLE = True
except ImportError:
    HF_DATASETS_AVAILABLE = False
    warnings.warn("Hugging Face datasets not installed. Install with: pip install datasets")


class DialogueDataLoader:
    """
    Handles loading and validation of dialogue summarization datasets from CSV files.
    
    This class provides robust data loading with validation, cleaning, and
    error handling for dialogue summarization tasks.
    
    Attributes:
        data_dir: Path to the directory containing data files
        required_train_columns: Required columns for training data
        required_test_columns: Required columns for test data
    """
    
    # Define expected column names
    REQUIRED_TRAIN_COLUMNS = ['id', 'dialogue', 'summary']
    REQUIRED_TEST_COLUMNS = ['id', 'dialogue']
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize the data loader.
        
        Args:
            data_dir: Directory containing the data files
            
        Raises:
            FileNotFoundError: If data directory does not exist
        """
        self.data_dir = Path(data_dir)
        
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
        
        self.required_train_columns = self.REQUIRED_TRAIN_COLUMNS
        self.required_test_columns = self.REQUIRED_TEST_COLUMNS
        
    def _validate_columns(
        self, 
        df: pd.DataFrame, 
        required_columns: List[str],
        filename: str
    ) -> None:
        """
        Validate that DataFrame contains all required columns.
        
        Args:
            df: DataFrame to validate
            required_columns: List of required column names
            filename: Name of file being validated (for error messages)
            
        Raises:
            ValueError: If required columns are missing
        """
        missing_columns = set(required_columns) - set(df.columns)
        
        if missing_columns:
            raise ValueError(
                f"Missing required columns in {filename}: {missing_columns}\n"
                f"Expected columns: {required_columns}\n"
                f"Found columns: {list(df.columns)}"
            )
    
    def _clean_dataframe(
        self, 
        df: pd.DataFrame, 
        is_training: bool = True
    ) -> pd.DataFrame:
        """
        Clean and validate DataFrame content.
        
        Args:
            df: DataFrame to clean
            is_training: Whether this is training data (expects summary column)
            
        Returns:
            Cleaned DataFrame
        """
        df = df.copy()
        
        # Remove duplicate IDs
        initial_count = len(df)
        df = df.drop_duplicates(subset=['id'], keep='first')
        duplicates_removed = initial_count - len(df)
        if duplicates_removed > 0:
            warnings.warn(f"Removed {duplicates_removed} duplicate IDs")
        
        # Convert columns to string type
        df['id'] = df['id'].astype(str)
        df['dialogue'] = df['dialogue'].fillna('').astype(str)
        
        if is_training and 'summary' in df.columns:
            df['summary'] = df['summary'].fillna('').astype(str)
        
        # Remove rows with empty dialogues
        empty_dialogue_mask = df['dialogue'].str.strip() == ''
        empty_dialogue_count = empty_dialogue_mask.sum()
        
        if empty_dialogue_count > 0:
            warnings.warn(
                f"Found {empty_dialogue_count} rows with empty dialogues. "
                f"These will be removed."
            )
            df = df[~empty_dialogue_mask]
        
        # For training data, warn about empty summaries but don't remove
        if is_training and 'summary' in df.columns:
            empty_summary_mask = df['summary'].str.strip() == ''
            empty_summary_count = empty_summary_mask.sum()
            
            if empty_summary_count > 0:
                warnings.warn(
                    f"Found {empty_summary_count} rows with empty summaries. "
                    f"Consider removing or handling these during preprocessing."
                )
        
        # Reset index after cleaning
        df = df.reset_index(drop=True)
        
        return df
    
    def load_csv(
        self, 
        filename: str,
        validate: bool = True,
        clean: bool = True,
        is_training: bool = True
    ) -> pd.DataFrame:
        """
        Load data from a CSV file with validation and cleaning.
        
        Args:
            filename: Name of the CSV file to load
            validate: Whether to validate required columns
            clean: Whether to clean and handle missing values
            is_training: Whether this is training data (expects summary column)
            
        Returns:
            DataFrame containing the loaded data
            
        Raises:
            FileNotFoundError: If file does not exist
            ValueError: If required columns are missing
        """
        filepath = self.data_dir / filename
        
        if not filepath.exists():
            raise FileNotFoundError(
                f"File not found: {filepath}\n"
                f"Please ensure the file exists in the data directory: {self.data_dir}"
            )
        
        # Load CSV
        try:
            df = pd.read_csv(filepath, encoding='utf-8')
        except UnicodeDecodeError:
            # Fallback to different encoding
            df = pd.read_csv(filepath, encoding='latin-1')
            warnings.warn(f"Loaded {filename} with latin-1 encoding")
        
        print(f"Loaded {len(df)} records from {filename}")
        
        # Validate columns
        if validate:
            required_cols = self.required_train_columns if is_training else self.required_test_columns
            self._validate_columns(df, required_cols, filename)
        
        # Clean data
        if clean:
            df = self._clean_dataframe(df, is_training=is_training)
            print(f"After cleaning: {len(df)} records")
        
        return df
    
    def load_train_data(
        self, 
        filename: str = "train.csv",
        validate: bool = True,
        clean: bool = True
    ) -> pd.DataFrame:
        """
        Load training dataset (includes summary column).
        
        Args:
            filename: Name of the training data file
            validate: Whether to validate required columns
            clean: Whether to clean and handle missing values
            
        Returns:
            DataFrame with columns: id, dialogue, summary
            
        Raises:
            FileNotFoundError: If file does not exist
            ValueError: If required columns are missing
        """
        print("\n" + "="*60)
        print("Loading Training Data")
        print("="*60)
        
        df = self.load_csv(
            filename, 
            validate=validate, 
            clean=clean, 
            is_training=True
        )
        
        # Additional validation for training data
        if clean:
            # Remove rows where summary is just whitespace
            original_len = len(df)
            df = df[df['summary'].str.strip().str.len() > 0]
            removed = original_len - len(df)
            if removed > 0:
                print(f"Removed {removed} rows with empty summaries")
        
        print(f"Final training samples: {len(df)}")
        print("="*60 + "\n")
        
        return df
    
    def load_test_data(
        self, 
        filename: str = "test.csv",
        validate: bool = True,
        clean: bool = True
    ) -> pd.DataFrame:
        """
        Load test dataset (no summary column expected).
        
        Args:
            filename: Name of the test data file
            validate: Whether to validate required columns
            clean: Whether to clean and handle missing values
            
        Returns:
            DataFrame with columns: id, dialogue
            
        Raises:
            FileNotFoundError: If file does not exist
            ValueError: If required columns are missing
        """
        print("\n" + "="*60)
        print("Loading Test Data")
        print("="*60)
        
        df = self.load_csv(
            filename, 
            validate=validate, 
            clean=clean, 
            is_training=False
        )
        
        print(f"Final test samples: {len(df)}")
        print("="*60 + "\n")
        
        return df
    
    def load_train_test_data(
        self, 
        train_file: str = "train.csv",
        test_file: str = "test.csv",
        validate: bool = True,
        clean: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load both training and test datasets.
        
        Args:
            train_file: Name of the training data file
            test_file: Name of the test data file
            validate: Whether to validate required columns
            clean: Whether to clean and handle missing values
            
        Returns:
            Tuple of (train_df, test_df)
            
        Raises:
            FileNotFoundError: If files do not exist
            ValueError: If required columns are missing
        """
        train_df = self.load_train_data(train_file, validate=validate, clean=clean)
        test_df = self.load_test_data(test_file, validate=validate, clean=clean)
        
        return train_df, test_df
    
    def split_train_validation(
        self,
        df: pd.DataFrame,
        val_split: float = 0.1,
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split training data into train and validation sets.
        
        Args:
            df: DataFrame to split
            val_split: Proportion of data for validation (0.0 to 1.0)
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple of (train_df, val_df)
            
        Raises:
            ValueError: If val_split is not between 0 and 1
        """
        if not 0 < val_split < 1:
            raise ValueError(f"val_split must be between 0 and 1, got {val_split}")
        
        # Shuffle and split
        df_shuffled = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
        split_idx = int(len(df_shuffled) * (1 - val_split))
        
        train_df = df_shuffled[:split_idx].reset_index(drop=True)
        val_df = df_shuffled[split_idx:].reset_index(drop=True)
        
        print(f"Split data: {len(train_df)} training, {len(val_df)} validation")
        
        return train_df, val_df
    
    def convert_to_hf_dataset(
        self, 
        train_df: pd.DataFrame,
        val_df: Optional[pd.DataFrame] = None,
        test_df: Optional[pd.DataFrame] = None
    ) -> 'DatasetDict':
        """
        Convert pandas DataFrames to Hugging Face DatasetDict.
        
        Args:
            train_df: Training data DataFrame
            val_df: Validation data DataFrame (optional)
            test_df: Test data DataFrame (optional)
            
        Returns:
            DatasetDict containing the datasets
            
        Raises:
            ImportError: If datasets library is not installed
        """
        if not HF_DATASETS_AVAILABLE:
            raise ImportError(
                "Hugging Face datasets library is required for this function. "
                "Install with: pip install datasets"
            )
        
        dataset_dict = {"train": Dataset.from_pandas(train_df, preserve_index=False)}
        
        if val_df is not None:
            dataset_dict["validation"] = Dataset.from_pandas(val_df, preserve_index=False)
        
        if test_df is not None:
            dataset_dict["test"] = Dataset.from_pandas(test_df, preserve_index=False)
        
        return DatasetDict(dataset_dict)
    
    def get_dataset_info(self, df: pd.DataFrame, name: str = "Dataset") -> Dict[str, Any]:
        """
        Get comprehensive statistics and information about the dataset.
        
        Args:
            df: DataFrame to analyze
            name: Name of the dataset for display purposes
            
        Returns:
            Dictionary containing dataset information
        """
        info = {
            "name": name,
            "num_samples": len(df),
            "num_features": len(df.columns),
            "columns": list(df.columns),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.to_dict().items()},
            "missing_values": df.isnull().sum().to_dict(),
            "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024 / 1024
        }
        
        # Add text length statistics if dialogue column exists
        if 'dialogue' in df.columns:
            dialogue_lengths = df['dialogue'].astype(str).str.split().str.len()
            info['dialogue_stats'] = {
                'mean_length': float(dialogue_lengths.mean()),
                'median_length': float(dialogue_lengths.median()),
                'min_length': int(dialogue_lengths.min()),
                'max_length': int(dialogue_lengths.max()),
                'std_length': float(dialogue_lengths.std())
            }
        
        # Add summary statistics if summary column exists
        if 'summary' in df.columns:
            summary_lengths = df['summary'].astype(str).str.split().str.len()
            info['summary_stats'] = {
                'mean_length': float(summary_lengths.mean()),
                'median_length': float(summary_lengths.median()),
                'min_length': int(summary_lengths.min()),
                'max_length': int(summary_lengths.max()),
                'std_length': float(summary_lengths.std())
            }
            
            # Compression ratio
            dialogue_lengths = df['dialogue'].astype(str).str.split().str.len()
            compression_ratio = summary_lengths / dialogue_lengths
            info['compression_ratio'] = {
                'mean': float(compression_ratio.mean()),
                'median': float(compression_ratio.median()),
                'std': float(compression_ratio.std())
            }
        
        return info
    
    def print_dataset_info(self, df: pd.DataFrame, name: str = "Dataset") -> None:
        """
        Print formatted dataset information.
        
        Args:
            df: DataFrame to analyze
            name: Name of the dataset for display
        """
        info = self.get_dataset_info(df, name)
        
        print("\n" + "="*60)
        print(f"{info['name']} Information")
        print("="*60)
        print(f"Samples: {info['num_samples']:,}")
        print(f"Features: {info['num_features']}")
        print(f"Columns: {', '.join(info['columns'])}")
        print(f"Memory Usage: {info['memory_usage_mb']:.2f} MB")
        
        if info['missing_values']:
            print("\nMissing Values:")
            for col, count in info['missing_values'].items():
                if count > 0:
                    print(f"  {col}: {count}")
        
        if 'dialogue_stats' in info:
            stats = info['dialogue_stats']
            print("\nDialogue Statistics (words):")
            print(f"  Mean: {stats['mean_length']:.1f}")
            print(f"  Median: {stats['median_length']:.1f}")
            print(f"  Range: [{stats['min_length']}, {stats['max_length']}]")
            print(f"  Std Dev: {stats['std_length']:.1f}")
        
        if 'summary_stats' in info:
            stats = info['summary_stats']
            print("\nSummary Statistics (words):")
            print(f"  Mean: {stats['mean_length']:.1f}")
            print(f"  Median: {stats['median_length']:.1f}")
            print(f"  Range: [{stats['min_length']}, {stats['max_length']}]")
            print(f"  Std Dev: {stats['std_length']:.1f}")
            
            ratio = info['compression_ratio']
            print("\nCompression Ratio (summary/dialogue):")
            print(f"  Mean: {ratio['mean']:.3f}")
            print(f"  Median: {ratio['median']:.3f}")
        
        print("="*60 + "\n")
    
    def save_dataframe(
        self,
        df: pd.DataFrame,
        filename: str,
        overwrite: bool = False
    ) -> None:
        """
        Save DataFrame to CSV file.
        
        Args:
            df: DataFrame to save
            filename: Name of output file
            overwrite: Whether to overwrite existing file
            
        Raises:
            FileExistsError: If file exists and overwrite is False
        """
        filepath = self.data_dir / filename
        
        if filepath.exists() and not overwrite:
            raise FileExistsError(
                f"File already exists: {filepath}\n"
                f"Set overwrite=True to replace it."
            )
        
        df.to_csv(filepath, index=False, encoding='utf-8')
        print(f"Saved {len(df)} records to {filename}")


def load_dialogue_data(
    data_dir: str = "data",
    train_file: str = "train.csv",
    test_file: str = "test.csv",
    validate: bool = True,
    clean: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convenience function to load training and test data.
    
    Args:
        data_dir: Directory containing the data files
        train_file: Name of the training data file
        test_file: Name of the test data file
        validate: Whether to validate required columns
        clean: Whether to clean and handle missing values
        
    Returns:
        Tuple of (train_df, test_df)
        
    Raises:
        FileNotFoundError: If files do not exist
        ValueError: If required columns are missing
    """
    loader = DialogueDataLoader(data_dir=data_dir)
    return loader.load_train_test_data(
        train_file=train_file,
        test_file=test_file,
        validate=validate,
        clean=clean
    )


if __name__ == "__main__":
    # Example usage and testing
    print("="*60)
    print("Dialogue Summarization Data Loader - Example Usage")
    print("="*60)
    
    try:
        # Initialize loader
        loader = DialogueDataLoader(data_dir="data")
        
        # Load training and test data
        train_df, test_df = loader.load_train_test_data()
        
        # Display dataset information
        loader.print_dataset_info(train_df, name="Training Data")
        loader.print_dataset_info(test_df, name="Test Data")
        
        # Show sample records
        print("\nSample Training Records:")
        print("-"*60)
        print(train_df.head(3))
        
        print("\nSample Test Records:")
        print("-"*60)
        print(test_df.head(3))
        
        # Optional: Split training data into train/validation
        print("\nSplitting training data into train/validation...")
        train_split, val_split = loader.split_train_validation(train_df, val_split=0.1)
        print(f"Train: {len(train_split)}, Validation: {len(val_split)}")
        
        # Optional: Convert to HuggingFace datasets
        if HF_DATASETS_AVAILABLE:
            print("\nConverting to HuggingFace DatasetDict...")
            dataset_dict = loader.convert_to_hf_dataset(
                train_df=train_split,
                val_df=val_split,
                test_df=test_df
            )
            print(f"Dataset splits: {list(dataset_dict.keys())}")
            print(dataset_dict)
        
        print("\n" + "="*60)
        print("Data loading completed successfully!")
        print("="*60)
        
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("Please ensure data files exist in the data/ directory")
    except ValueError as e:
        print(f"\nValidation Error: {e}")
    except Exception as e:
        print(f"\nUnexpected Error: {e}")
        import traceback
        traceback.print_exc()
