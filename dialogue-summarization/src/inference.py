"""
Inference Module for Dialogue Summarization

This module handles cloud-safe inference/prediction for dialogue summarization
using fine-tuned Hugging Face transformer models.

Key Features:
- Load trained models and tokenizers from models/ directory
- GPU/CPU device auto-detection and support
- Preprocessing integration with existing utilities
- Batch inference for efficient processing
- Clean, non-empty summary generation
- Competition-ready CSV output format (id,summary)

The module uses torch.no_grad() and evaluation mode for memory efficiency
and proper inference behavior.
"""

import os
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from typing import List, Tuple

# Import preprocessing utilities
from data_loader import load_test_data
from preprocess import preprocess_dataframe


# Constants for model inference
MAX_INPUT_LENGTH = 512  # Maximum length for input dialogues
NUM_BEAMS = 4  # Number of beams for beam search
LENGTH_PENALTY = 1.0  # Length penalty for generation (neutral)
NO_REPEAT_NGRAM_SIZE = 3  # Prevent repetition of n-grams


def load_model(model_path: str, device: str = 'cpu') -> Tuple[AutoModelForSeq2SeqLM, AutoTokenizer]:
    """
    Load a trained model and tokenizer from checkpoint directory.
    
    This function loads a fine-tuned seq2seq model (T5, BART, etc.) and its
    associated tokenizer from a saved checkpoint. Supports both single model
    files and Hugging Face format directories.
    
    Args:
        model_path (str): Path to the model checkpoint directory.
                         Should contain config.json, pytorch_model.bin, and tokenizer files.
        device (str): Device to load model on ('cuda' or 'cpu').
                     Automatically detected if not specified.
        
    Returns:
        tuple: (model, tokenizer) - The loaded model and tokenizer instances
        
    Raises:
        FileNotFoundError: If model files are not found at the specified path
        ValueError: If model cannot be loaded properly
        
    Examples:
        >>> model, tokenizer = load_model('../models/dialogue-summarizer/', 'cuda')
        >>> print(f"Model loaded on {device}")
    """
    print(f"Loading model from: {model_path}")
    
    # Validate path exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path does not exist: {model_path}")
    
    try:
        # Load tokenizer from pretrained checkpoint
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Load model for seq2seq generation
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        
        # Move model to specified device
        model = model.to(device)
        
        # Set model to evaluation mode (disables dropout, etc.)
        model.eval()
        
        print(f"✓ Model loaded successfully")
        print(f"  Model type: {model.config.model_type}")
        print(f"  Device: {device}")
        print(f"  Vocab size: {tokenizer.vocab_size}")
        
        return model, tokenizer
        
    except Exception as e:
        raise ValueError(f"Failed to load model from '{model_path}': {str(e)}")


def generate_summary(
    model: AutoModelForSeq2SeqLM,
    tokenizer: AutoTokenizer,
    dialogue: str,
    max_length: int = 150,
    min_length: int = 20,
    device: str = 'cpu'
) -> str:
    """
    Generate a concise summary for a single dialogue using the trained model.
    
    This function tokenizes the input dialogue, generates a summary using the
    seq2seq model, and decodes the result into clean, readable text. Uses
    torch.no_grad() for memory efficiency during inference.
    
    Args:
        model: Fine-tuned seq2seq model instance
        tokenizer: Tokenizer instance matching the model
        dialogue (str): Input dialogue text to summarize
        max_length (int): Maximum length of generated summary (default: 150)
        min_length (int): Minimum length of generated summary (default: 20)
        device (str): Device for inference ('cuda' or 'cpu')
        
    Returns:
        str: Generated summary text, cleaned and readable
        
    Examples:
        >>> dialogue = "John: Hi! How are you?\\nMary: I'm great, thanks!"
        >>> summary = generate_summary(model, tokenizer, dialogue, device='cuda')
        >>> print(summary)
        'John and Mary exchanged greetings.'
    """
    # Set model to evaluation mode
    model.eval()
    
    # Use no_grad context for inference (no gradient computation needed)
    with torch.no_grad():
        # Tokenize input dialogue
        inputs = tokenizer(
            dialogue,
            max_length=MAX_INPUT_LENGTH,
            truncation=True,
            padding=False,
            return_tensors='pt'
        )
        
        # Move input tensors to the same device as model
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate summary using model.generate()
        # Use beam search for better quality
        output_ids = model.generate(
            inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_length=max_length,
            min_length=min_length,
            num_beams=NUM_BEAMS,
            length_penalty=LENGTH_PENALTY,
            early_stopping=True,  # Stop when all beams finish
            no_repeat_ngram_size=NO_REPEAT_NGRAM_SIZE,
        )
        
        # Decode generated tokens to text
        summary = tokenizer.decode(
            output_ids[0],
            skip_special_tokens=True,  # Remove <pad>, </s>, etc.
            clean_up_tokenization_spaces=True  # Clean up spacing
        )
        
        # Post-process: strip whitespace and ensure non-empty
        summary = summary.strip()
        
        # Ensure summary is not empty (fallback)
        if not summary:
            summary = "Summary not available."
    
    return summary


def batch_inference(
    model: AutoModelForSeq2SeqLM,
    tokenizer: AutoTokenizer,
    dialogues: List[str],
    max_length: int = 150,
    min_length: int = 20,
    batch_size: int = 8,
    device: str = 'cpu'
) -> List[str]:
    """
    Generate summaries for multiple dialogues in batches for efficiency.
    
    Processes dialogues in batches to optimize GPU/CPU usage and reduce
    inference time. Uses torch.no_grad() for memory efficiency.
    
    Args:
        model: Fine-tuned seq2seq model instance
        tokenizer: Tokenizer instance matching the model
        dialogues (list): List of dialogue texts to summarize
        max_length (int): Maximum length of generated summaries (default: 150)
        min_length (int): Minimum length of generated summaries (default: 20)
        batch_size (int): Number of dialogues to process at once (default: 8)
        device (str): Device for inference ('cuda' or 'cpu')
        
    Returns:
        list: List of generated summaries in the same order as input dialogues
        
    Examples:
        >>> dialogues = ["Dialogue 1...", "Dialogue 2...", "Dialogue 3..."]
        >>> summaries = batch_inference(model, tokenizer, dialogues, batch_size=2)
        >>> print(f"Generated {len(summaries)} summaries")
    """
    summaries = []
    
    # Set model to evaluation mode
    model.eval()
    
    print(f"Processing {len(dialogues)} dialogues in batches of {batch_size}...")
    
    # Process dialogues in batches for efficiency
    for i in range(0, len(dialogues), batch_size):
        batch = dialogues[i:i+batch_size]
        
        # Use no_grad context for inference
        with torch.no_grad():
            # Tokenize batch of dialogues
            # Dynamic padding to longest in batch for efficiency
            inputs = tokenizer(
                batch,
                max_length=MAX_INPUT_LENGTH,
                truncation=True,
                padding=True,  # Pad to longest in batch
                return_tensors='pt'
            )
            
            # Move input tensors to device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Generate summaries for batch
            output_ids = model.generate(
                inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_length=max_length,
                min_length=min_length,
                num_beams=NUM_BEAMS,
                length_penalty=LENGTH_PENALTY,
                early_stopping=True,
                no_repeat_ngram_size=NO_REPEAT_NGRAM_SIZE,
            )
            
            # Decode each summary in the batch
            batch_summaries = tokenizer.batch_decode(
                output_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            
            # Post-process summaries
            for summary in batch_summaries:
                summary = summary.strip()
                # Ensure non-empty summaries
                if not summary:
                    summary = "Summary not available."
                summaries.append(summary)
        
        # Progress indicator
        if (i // batch_size + 1) % 10 == 0 or i + batch_size >= len(dialogues):
            print(f"  Processed {min(i + batch_size, len(dialogues))}/{len(dialogues)} dialogues")
    
    print(f"✓ Completed inference for {len(summaries)} dialogues")
    
    return summaries


def run_inference(
    model_path: str,
    test_data_path: str,
    output_path: str,
    batch_size: int = 8,
    max_length: int = 150,
    min_length: int = 20
) -> None:
    """
    Complete inference pipeline: load model, process test data, generate predictions.
    
    This is the main entry point for the inference pipeline. It orchestrates:
    1. Device detection (GPU/CPU)
    2. Model and tokenizer loading
    3. Test data loading and preprocessing
    4. Batch inference
    5. Saving predictions to CSV
    
    The output CSV has exactly two columns: id,summary (ready for submission).
    
    Args:
        model_path (str): Path to trained model checkpoint directory
        test_data_path (str): Path to test CSV file with dialogues
        output_path (str): Path to save predictions CSV file
        batch_size (int): Batch size for inference (default: 8)
        max_length (int): Maximum summary length (default: 150)
        min_length (int): Minimum summary length (default: 20)
        
    Raises:
        FileNotFoundError: If model or data files are not found
        ValueError: If data format is invalid
        
    Examples:
        >>> run_inference(
        ...     model_path='../models/dialogue-summarizer/',
        ...     test_data_path='../data/samsum_test.csv',
        ...     output_path='../data/predictions.csv'
        ... )
    """
    print("=" * 70)
    print("DIALOGUE SUMMARIZATION INFERENCE PIPELINE")
    print("=" * 70)
    print()
    
    # Step 1: Detect and configure device (GPU if available, otherwise CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("  Running on CPU (GPU not available)")
    print()
    
    # Step 2: Load trained model and tokenizer
    print("Loading model and tokenizer...")
    print("-" * 70)
    model, tokenizer = load_model(model_path, device=str(device))
    print()
    
    # Step 3: Load test data using load_test_data()
    print("Loading test data...")
    print("-" * 70)
    test_df = load_test_data(test_data_path)
    print()
    
    # Step 4: Apply preprocess_dataframe() before inference
    print("Preprocessing data...")
    print("-" * 70)
    test_df = preprocess_dataframe(test_df)
    print(f"✓ Preprocessed {len(test_df)} test samples")
    print()
    
    # Extract dialogues and IDs
    dialogues = test_df['dialogue'].tolist()
    dialogue_ids = test_df['id'].tolist()
    
    # Step 5: Generate summaries using batch inference
    print("Generating summaries...")
    print("-" * 70)
    summaries = batch_inference(
        model=model,
        tokenizer=tokenizer,
        dialogues=dialogues,
        max_length=max_length,
        min_length=min_length,
        batch_size=batch_size,
        device=str(device)
    )
    print()
    
    # Step 6: Save predictions to CSV with exactly two columns: id,summary
    print("Saving predictions...")
    print("-" * 70)
    predictions_df = pd.DataFrame({
        'id': dialogue_ids,
        'summary': summaries
    })
    
    # Ensure output directory exists (if output_path includes a directory)
    # When output_dir is empty string, file will be saved in current directory
    output_dir = os.path.dirname(output_path)
    if output_dir:  # Only create directory if output_path includes a directory component
        os.makedirs(output_dir, exist_ok=True)
    
    # Save to CSV
    predictions_df.to_csv(output_path, index=False)
    
    print(f"✓ Predictions saved to: {output_path}")
    print(f"  Format: CSV with columns 'id' and 'summary'")
    print(f"  Total predictions: {len(predictions_df)}")
    print()
    
    # Display sample predictions
    print("Sample predictions:")
    print("-" * 70)
    for i in range(min(3, len(predictions_df))):
        print(f"ID: {predictions_df['id'].iloc[i]}")
        print(f"Summary: {predictions_df['summary'].iloc[i]}")
        print()
    
    print("=" * 70)
    print("INFERENCE COMPLETE")
    print("=" * 70)
    print(f"Output file ready for submission: {output_path}")


if __name__ == "__main__":
    """
    Main execution block for inference.
    
    Run this script to generate predictions for test data using a trained model.
    Adjust paths and parameters based on your setup.
    
    Usage:
        python inference.py
    """
    
    # Define paths relative to src directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Configuration
    config = {
        'model_path': os.path.join(base_dir, '..', 'models', 'dialogue-summarizer'),
        'test_data_path': os.path.join(base_dir, '..', 'data', 'samsum_test.csv'),
        'output_path': os.path.join(base_dir, '..', 'data', 'predictions.csv'),
        'batch_size': 8,  # Adjust based on available memory
        'max_length': 150,  # Maximum summary length
        'min_length': 20,  # Minimum summary length
    }
    
    # Run inference pipeline
    run_inference(**config)
