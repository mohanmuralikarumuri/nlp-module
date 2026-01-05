"""
Preprocessing Module

This module contains functions for preprocessing dialogue data before training,
including text cleaning, speaker preservation, multi-turn dialogue formatting,
and tokenization for transformer-based summarization models (T5, BART, etc.).
"""

import re
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
import warnings

try:
    from transformers import PreTrainedTokenizer
    from datasets import Dataset
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    PreTrainedTokenizer = None
    Dataset = None


class DialoguePreprocessor:
    """
    Handles preprocessing of dialogue-summarization data.
    
    This class provides comprehensive text cleaning, speaker preservation,
    multi-turn dialogue formatting, and tokenization for transformer models.
    Designed to work with chat-style dialogues for T5, BART, and similar models.
    """
    
    def __init__(
        self, 
        tokenizer: Optional['PreTrainedTokenizer'] = None,
        max_input_length: int = 512,
        max_target_length: int = 128,
        dialogue_format: str = "standard"
    ):
        """
        Initialize the preprocessor.
        
        Args:
            tokenizer: Hugging Face tokenizer (optional for text-only preprocessing)
            max_input_length: Maximum length for input sequences
            max_target_length: Maximum length for target summaries
            dialogue_format: Format style ('standard', 'turns', 'compact')
        """
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
        self.dialogue_format = dialogue_format
    
    def clean_text(self, text: str, preserve_newlines: bool = False) -> str:
        """
        Clean and normalize text while preserving dialogue structure.
        
        Args:
            text: Input text to clean
            preserve_newlines: Whether to preserve newline characters
            
        Returns:
            Cleaned text
        """
        if not isinstance(text, str):
            return ""
        
        # Replace common chat artifacts
        text = text.replace('\\r\\n', '\n').replace('\\n', '\n')
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+', '[URL]', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '[EMAIL]', text)
        
        # Normalize punctuation spacing
        text = re.sub(r'\s+([.,!?;:])', r'\1', text)
        text = re.sub(r'([.,!?;:])([^\s.,!?;:])', r'\1 \2', text)
        
        # Remove multiple consecutive punctuation (except ...)
        text = re.sub(r'([!?]){2,}', r'\1', text)
        
        # Handle extra whitespace
        if preserve_newlines:
            # Preserve single newlines, remove multiple
            text = re.sub(r'\n{3,}', '\n\n', text)
            text = re.sub(r'[^\S\n]+', ' ', text)  # Remove spaces/tabs but keep newlines
        else:
            text = re.sub(r'\s+', ' ', text)
        
        # Trim
        text = text.strip()
        
        return text
    
    def normalize_speakers(self, text: str) -> str:
        """
        Normalize speaker name formatting in dialogues.
        
        Args:
            text: Dialogue text with speaker names
            
        Returns:
            Text with normalized speaker formatting
        """
        # Normalize "Person A:" or "PersonA:" patterns
        text = re.sub(r'\bPerson\s*([A-Z])\s*:', r'Person \1:', text)
        
        # Normalize general "Name:" patterns (ensure space after colon)
        text = re.sub(r'([A-Za-z][A-Za-z0-9_-]*):\s*', r'\1: ', text)
        
        # Remove extra spaces after speaker names
        text = re.sub(r':\s{2,}', ': ', text)
        
        return text
    
    def parse_dialogue_turns(self, dialogue: str) -> List[Tuple[str, str]]:
        """
        Parse multi-turn dialogue into (speaker, message) pairs.
        
        Args:
            dialogue: Raw dialogue text
            
        Returns:
            List of (speaker_name, message) tuples
        """
        turns = []
        
        # Split by newlines first
        lines = dialogue.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Match "Speaker: message" pattern
            match = re.match(r'^([^:]+):\s*(.+)$', line)
            if match:
                speaker = match.group(1).strip()
                message = match.group(2).strip()
                if speaker and message:
                    turns.append((speaker, message))
            else:
                # If no speaker found, append to previous message if exists
                if turns:
                    prev_speaker, prev_message = turns[-1]
                    turns[-1] = (prev_speaker, prev_message + ' ' + line)
        
        return turns
    
    def format_dialogue(self, dialogue: str, format_style: Optional[str] = None) -> str:
        """
        Format multi-turn dialogue for model input.
        
        Args:
            dialogue: Raw dialogue text
            format_style: Format style ('standard', 'turns', 'compact')
                         If None, uses self.dialogue_format
            
        Returns:
            Formatted dialogue string
        """
        format_style = format_style or self.dialogue_format
        
        # Clean and normalize
        dialogue = self.clean_text(dialogue, preserve_newlines=True)
        dialogue = self.normalize_speakers(dialogue)
        
        if format_style == 'turns':
            # Parse into turns and format with separators
            turns = self.parse_dialogue_turns(dialogue)
            if turns:
                formatted = ' '.join([f"{speaker}: {msg}" for speaker, msg in turns])
            else:
                formatted = dialogue.replace('\n', ' ')
        
        elif format_style == 'compact':
            # Compact format without explicit speakers
            turns = self.parse_dialogue_turns(dialogue)
            if turns:
                formatted = ' '.join([msg for _, msg in turns])
            else:
                formatted = dialogue.replace('\n', ' ')
        
        else:  # 'standard'
            # Keep original structure but clean
            formatted = dialogue.replace('\n', ' ')
        
        # Final cleanup
        formatted = re.sub(r'\s+', ' ', formatted).strip()
        
        return formatted
    
    def preprocess_dataframe(
        self,
        df: pd.DataFrame,
        dialogue_col: str = 'dialogue',
        summary_col: str = 'summary',
        id_col: str = 'id'
    ) -> pd.DataFrame:
        """
        Preprocess a pandas DataFrame with dialogue data.
        
        Args:
            df: Input DataFrame
            dialogue_col: Name of dialogue column
            summary_col: Name of summary column (optional)
            id_col: Name of ID column
            
        Returns:
            DataFrame with cleaned and formatted text
        """
        df_processed = df.copy()
        
        # Validate required columns
        if dialogue_col not in df.columns:
            raise ValueError(f"Dialogue column '{dialogue_col}' not found in DataFrame")
        
        # Process dialogues
        print(f"Processing {len(df)} dialogues...")
        df_processed[dialogue_col] = df[dialogue_col].apply(
            lambda x: self.format_dialogue(str(x)) if pd.notna(x) else ""
        )
        
        # Process summaries if present
        if summary_col in df.columns:
            print(f"Processing summaries...")
            df_processed[summary_col] = df[summary_col].apply(
                lambda x: self.clean_text(str(x)) if pd.notna(x) else ""
            )
        
        # Remove empty dialogues
        empty_mask = df_processed[dialogue_col].str.strip() == ''
        empty_count = empty_mask.sum()
        
        if empty_count > 0:
            warnings.warn(f"Removing {empty_count} rows with empty processed dialogues")
            df_processed = df_processed[~empty_mask].reset_index(drop=True)
        
        print(f"Preprocessing complete: {len(df_processed)} dialogues")
        
        return df_processed
    
    def preprocess_function(self, examples: Dict[str, List]) -> Dict[str, Any]:
        """
        Preprocess a batch of examples for training.
        
        This function is designed to be used with datasets.map().
        Formats dialogues and tokenizes for model training.
        
        Args:
            examples: Dictionary containing batches of dialogues and summaries
            
        Returns:
            Dictionary with tokenized inputs and labels
            
        Raises:
            ValueError: If tokenizer is not set
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer must be set for tokenization. Initialize with a tokenizer.")
        
        # Format and clean dialogues
        inputs = [self.format_dialogue(dialogue) for dialogue in examples.get("dialogue", [])]
        targets = [self.clean_text(summary) for summary in examples.get("summary", [])]
        
        # Tokenize inputs
        model_inputs = self.tokenizer(
            inputs,
            max_length=self.max_input_length,
            truncation=True,
            padding="max_length"
        )
        
        # Tokenize targets
        labels = self.tokenizer(
            targets,
            max_length=self.max_target_length,
            truncation=True,
            padding="max_length"
        )
        
        model_inputs["labels"] = labels["input_ids"]
        
        return model_inputs
    
    def preprocess_dataset(
        self, 
        dataset: 'Dataset',
        batched: bool = True,
        num_proc: Optional[int] = None
    ) -> 'Dataset':
        """
        Apply preprocessing to an entire dataset.
        
        Args:
            dataset: Dataset to preprocess
            batched: Whether to process in batches
            num_proc: Number of processes for parallel processing
            
        Returns:
            Preprocessed dataset
            
        Raises:
            ImportError: If datasets library is not installed
        """
        if not HF_AVAILABLE:
            raise ImportError("datasets library required. Install with: pip install datasets")
        
        processed_dataset = dataset.map(
            self.preprocess_function,
            batched=batched,
            num_proc=num_proc,
            remove_columns=dataset.column_names
        )
        
        return processed_dataset
    
    def prepare_for_inference(self, dialogue: str) -> Dict[str, Any]:
        """
        Prepare a single dialogue for inference.
        
        Args:
            dialogue: Input dialogue text
            
        Returns:
            Tokenized inputs ready for model
            
        Raises:
            ValueError: If tokenizer is not set
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer must be set for tokenization. Initialize with a tokenizer.")
        
        # Format dialogue
        formatted_dialogue = self.format_dialogue(dialogue)
        
        inputs = self.tokenizer(
            formatted_dialogue,
            max_length=self.max_input_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        
        return inputs
    
    def get_dialogue_stats(self, df: pd.DataFrame, dialogue_col: str = 'dialogue') -> Dict[str, Any]:
        """
        Get statistics about processed dialogues.
        
        Args:
            df: DataFrame with dialogue data
            dialogue_col: Name of dialogue column
            
        Returns:
            Dictionary with statistics
        """
        stats = {}
        
        if dialogue_col in df.columns:
            # Word counts
            word_counts = df[dialogue_col].str.split().str.len()
            stats['dialogue'] = {
                'mean_words': float(word_counts.mean()),
                'median_words': float(word_counts.median()),
                'min_words': int(word_counts.min()),
                'max_words': int(word_counts.max()),
                'std_words': float(word_counts.std())
            }
            
            # Speaker detection
            has_speakers = df[dialogue_col].str.contains(':', regex=False)
            stats['speaker_info'] = {
                'dialogues_with_speakers': int(has_speakers.sum()),
                'percentage_with_speakers': float(has_speakers.mean() * 100)
            }
        
        return stats


def split_dataset(
    dataset: 'Dataset', 
    train_size: float = 0.8,
    val_size: float = 0.1,
    test_size: float = 0.1,
    seed: int = 42
) -> Dict[str, 'Dataset']:
    """
    Split a dataset into train, validation, and test sets.
    
    Args:
        dataset: Dataset to split
        train_size: Proportion for training
        val_size: Proportion for validation
        test_size: Proportion for testing
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary with 'train', 'validation', and 'test' splits
        
    Raises:
        ImportError: If datasets library is not installed
    """
    if not HF_AVAILABLE:
        raise ImportError("datasets library required. Install with: pip install datasets")
    
    assert abs(train_size + val_size + test_size - 1.0) < 1e-6, \
        "Split sizes must sum to 1.0"
    
    # First split: separate test set
    train_val_test = dataset.train_test_split(
        test_size=test_size, 
        seed=seed
    )
    
    # Second split: separate validation from train
    train_val = train_val_test["train"].train_test_split(
        test_size=val_size / (train_size + val_size),
        seed=seed
    )
    
    return {
        "train": train_val["train"],
        "validation": train_val["test"],
        "test": train_val_test["test"]
    }


def preprocess_dialogue_data(
    df: pd.DataFrame,
    dialogue_col: str = 'dialogue',
    summary_col: str = 'summary',
    format_style: str = 'standard'
) -> pd.DataFrame:
    """
    Convenience function to preprocess dialogue DataFrame.
    
    Args:
        df: Input DataFrame with dialogue data
        dialogue_col: Name of dialogue column
        summary_col: Name of summary column
        format_style: Dialogue format style ('standard', 'turns', 'compact')
        
    Returns:
        Preprocessed DataFrame
    """
    preprocessor = DialoguePreprocessor(dialogue_format=format_style)
    return preprocessor.preprocess_dataframe(df, dialogue_col, summary_col)


if __name__ == "__main__":
    # Example usage
    print("="*60)
    print("Dialogue Preprocessor - Example Usage")
    print("="*60)
    
    # Example 1: Text cleaning and formatting
    print("\n1. Text Cleaning and Formatting")
    print("-"*60)
    
    preprocessor = DialoguePreprocessor(dialogue_format='standard')
    
    sample_dialogue = """Person A: Hello, how are you?
 Person B: I'm doing great, thanks!!!
 Person A: That's wonderful to hear."""
    
    print("Original:")
    print(sample_dialogue)
    
    formatted = preprocessor.format_dialogue(sample_dialogue)
    print("\nFormatted:")
    print(formatted)
    
    # Example 2: Parse dialogue turns
    print("\n\n2. Parsing Dialogue Turns")
    print("-"*60)
    
    turns = preprocessor.parse_dialogue_turns(sample_dialogue)
    print(f"Found {len(turns)} turns:")
    for i, (speaker, message) in enumerate(turns, 1):
        print(f"  Turn {i}: {speaker} -> {message}")
    
    # Example 3: DataFrame preprocessing
    print("\n\n3. DataFrame Preprocessing")
    print("-"*60)
    
    # Create sample DataFrame
    sample_data = {
        'id': ['1', '2', '3'],
        'dialogue': [
            "Alice: Hi there!\nBob: Hello! How are you?",
            "John: What's the weather like?\nMary: It's sunny today.",
            "Tom:  Can you help me?   \nSarah:    Of course!!!"
        ],
        'summary': [
            "Alice and Bob greet each other.",
            "John asks about weather, Mary says it's sunny.",
            "Tom asks for help, Sarah agrees."
        ]
    }
    
    df = pd.DataFrame(sample_data)
    print("\nOriginal DataFrame:")
    print(df[['dialogue']].head())
    
    # Preprocess
    df_processed = preprocessor.preprocess_dataframe(df)
    print("\nProcessed DataFrame:")
    print(df_processed[['dialogue']].head())
    
    # Get statistics
    print("\n\n4. Dialogue Statistics")
    print("-"*60)
    stats = preprocessor.get_dialogue_stats(df_processed)
    print(f"Mean words per dialogue: {stats['dialogue']['mean_words']:.1f}")
    print(f"Dialogues with speakers: {stats['speaker_info']['percentage_with_speakers']:.1f}%")
    
    # Example 4: Tokenization (requires tokenizer)
    print("\n\n5. Tokenization Example")
    print("-"*60)
    
    try:
        from transformers import AutoTokenizer
        
        tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")
        preprocessor_with_tokenizer = DialoguePreprocessor(
            tokenizer=tokenizer,
            dialogue_format='standard'
        )
        
        # Prepare for inference
        inputs = preprocessor_with_tokenizer.prepare_for_inference(sample_dialogue)
        print(f"Tokenized input shape: {inputs['input_ids'].shape}")
        print(f"Number of tokens: {inputs['input_ids'].shape[1]}")
        
    except ImportError:
        print("Transformers library not installed. Skipping tokenization example.")
    
    print("\n" + "="*60)
    print("Examples completed!")
    print("="*60)
