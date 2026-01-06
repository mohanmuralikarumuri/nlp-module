"""
Preprocessing Module for Dialogue Summarization

This module contains functions for preprocessing dialogues and summaries.
Designed for cloud environments and transformer model training.

Key functionality includes:
- Text cleaning and normalization
- Dialogue formatting and structure extraction
- Handling special characters and encoding issues
- Data validation and quality checks
- Memory-efficient DataFrame operations
"""

import re
from typing import Optional
import pandas as pd


def remove_extra_spaces(text: str) -> str:
    """
    Remove extra whitespace while preserving text structure.
    
    Removes multiple consecutive spaces but preserves single spaces
    between words and maintains the overall text structure.
    
    Args:
        text (str): Text with potential extra spaces
        
    Returns:
        str: Text with normalized spacing
        
    Examples:
        >>> remove_extra_spaces("Hello    world")
        'Hello world'
        >>> remove_extra_spaces("Text  with   extra    spaces")
        'Text with extra spaces'
    """
    # Replace multiple spaces with single space
    text = re.sub(r' +', ' ', text)
    # Remove spaces at start of lines but keep line structure
    text = re.sub(r'^ +', '', text, flags=re.MULTILINE)
    # Remove spaces at end of lines
    text = re.sub(r' +$', '', text, flags=re.MULTILINE)
    return text


def normalize_newlines(text: str) -> str:
    """
    Normalize repeated newlines to improve readability.
    
    Reduces multiple consecutive newlines to maximum of two (paragraph break)
    while preserving dialogue structure and conversation flow.
    
    Args:
        text (str): Text with potential repeated newlines
        
    Returns:
        str: Text with normalized newlines
        
    Examples:
        >>> normalize_newlines("Line1\\n\\n\\n\\nLine2")
        'Line1\\n\\nLine2'
    """
    # Replace 3 or more newlines with exactly 2 (paragraph break)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text


def remove_special_characters(text: str) -> str:
    """
    Remove unnecessary special characters without losing meaning.
    
    Removes problematic special characters while preserving:
    - Punctuation essential for meaning (.,!?;:'"-)
    - Apostrophes in contractions
    - Speaker names and identifiers
    - Conversation structure markers
    
    Args:
        text (str): Text with potential problematic special characters
        
    Returns:
        str: Cleaned text with meaningful characters preserved
        
    Examples:
        >>> remove_special_characters("Hello! How are you?")
        'Hello! How are you?'
        >>> remove_special_characters("Test@#$%text")
        'Test text'
    """
    # Keep: letters, numbers, spaces, newlines, and essential punctuation
    # Essential punctuation: . , ! ? ; : ' " - ( )
    # Also preserve colon for speaker names (e.g., "John: Hello")
    text = re.sub(r'[^a-zA-Z0-9\s\n.,!?;:\'"()\\-]', ' ', text)
    # Clean up any multiple spaces created by removal
    text = re.sub(r' +', ' ', text)
    return text


def format_dialogue_for_transformer(dialogue: str) -> str:
    """
    Convert multi-turn dialogue into formatted string for transformer models.
    
    Preserves speaker names, conversation turns, and context while
    creating a clean, consistent format suitable for model input.
    
    Args:
        dialogue (str): Raw multi-turn dialogue text
        
    Returns:
        str: Formatted dialogue ready for tokenization
        
    Examples:
        >>> text = "John: Hi there\\nMary: Hello\\nJohn: How are you?"
        >>> format_dialogue_for_transformer(text)
        'John: Hi there\\nMary: Hello\\nJohn: How are you?'
    """
    # Apply cleaning steps in order
    dialogue = remove_extra_spaces(dialogue)
    dialogue = normalize_newlines(dialogue)
    dialogue = remove_special_characters(dialogue)
    
    # Final cleanup: strip leading/trailing whitespace
    dialogue = dialogue.strip()
    
    return dialogue


def clean_text(text: str) -> str:
    """
    Clean and normalize input text (backward compatibility function).
    
    Args:
        text (str): Raw text to be cleaned
        
    Returns:
        str: Cleaned and normalized text
    """
    # Remove extra whitespaces
    text = re.sub(r'\s+', ' ', text)
    # Strip leading/trailing whitespace
    text = text.strip()
    return text


def preprocess_dialogue(dialogue: str) -> str:
    """
    Preprocess dialogue text for model input.
    
    Args:
        dialogue (str): Raw dialogue text
        
    Returns:
        str: Preprocessed dialogue ready for tokenization
    """
    # Use the new comprehensive formatting function
    return format_dialogue_for_transformer(dialogue)


def preprocess_summary(summary: str) -> str:
    """
    Preprocess summary text for model training.
    
    Args:
        summary (str): Raw summary text
        
    Returns:
        str: Preprocessed summary
    """
    # Clean the summary text
    summary = clean_text(summary)
    return summary


def validate_data(dialogue: str, summary: Optional[str] = None) -> bool:
    """
    Validate dialogue and summary data for quality issues.
    
    Args:
        dialogue (str): Dialogue text to validate
        summary (str, optional): Summary text to validate
        
    Returns:
        bool: True if data is valid, False otherwise
    """
    # Check if dialogue is not empty
    if not dialogue or len(dialogue.strip()) == 0:
        return False
    
    # Check summary if provided
    if summary is not None:
        if not summary or len(summary.strip()) == 0:
            return False
    
    return True


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess pandas DataFrame for dialogue summarization.
    
    This is the main entry point for preprocessing. Accepts a DataFrame
    from the data loader and returns a cleaned DataFrame ready for model
    training or inference.
    
    Memory-efficient: processes columns in-place when possible and uses
    vectorized pandas operations for cloud environments.
    
    Args:
        df (pd.DataFrame): DataFrame with columns 'id', 'dialogue', 
                          and optionally 'summary'
        
    Returns:
        pd.DataFrame: Cleaned DataFrame with preprocessed text
        
    Raises:
        ValueError: If required columns are missing
        
    Examples:
        >>> import pandas as pd
        >>> df = pd.DataFrame({
        ...     'id': [1, 2],
        ...     'dialogue': ['John: Hi\\n\\nMary: Hello', 'A: Test\\nB: Reply'],
        ...     'summary': ['Greeting', 'Conversation']
        ... })
        >>> cleaned_df = preprocess_dataframe(df)
        >>> len(cleaned_df) == 2
        True
    """
    # Validate required columns
    if 'id' not in df.columns or 'dialogue' not in df.columns:
        raise ValueError("DataFrame must contain 'id' and 'dialogue' columns")
    
    # Create a copy to avoid modifying original DataFrame
    df_clean = df.copy()
    
    # Preprocess dialogue column (required)
    # Handle null values before processing
    # Use vectorized operation for efficiency
    df_clean['dialogue'] = df_clean['dialogue'].fillna('').apply(
        lambda x: format_dialogue_for_transformer(str(x)) if str(x).strip() else x
    )
    
    # Preprocess summary column if it exists (optional)
    if 'summary' in df_clean.columns:
        # Process summary, handling null values properly
        df_clean['summary'] = df_clean['summary'].apply(
            lambda x: clean_text(str(x)) if pd.notna(x) and str(x).strip() != '' else x
        )
    
    return df_clean


def preprocess_dataset(data_path: str, output_path: str) -> None:
    """
    Preprocess entire dataset from CSV and save to file.
    
    This function orchestrates the complete preprocessing pipeline:
    loads data, applies preprocessing, and saves results.
    
    Args:
        data_path (str): Path to input CSV file
        output_path (str): Path to save preprocessed data
        
    Raises:
        FileNotFoundError: If input file doesn't exist
        ValueError: If data format is invalid
    """
    # Load the data
    df = pd.read_csv(data_path)
    
    # Preprocess the dataframe
    df_processed = preprocess_dataframe(df)
    
    # Save preprocessed data
    df_processed.to_csv(output_path, index=False)
    
    print(f"Preprocessed {len(df_processed)} samples")
    print(f"Saved to: {output_path}")
