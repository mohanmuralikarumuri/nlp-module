"""
Training Module for Dialogue Summarization

This module handles the training loop for dialogue summarization models using
Hugging Face Transformers. Designed for cloud-safe execution with GPU/CPU support.

Key components include:
- Model initialization and configuration
- Data loading and preprocessing integration
- Training loop using Hugging Face Trainer API
- Checkpoint saving and model persistence
- Training metrics logging and monitoring
"""

import os
from typing import Dict, Optional
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
)
import pandas as pd

# Import data loading and preprocessing functions
from data_loader import load_train_data
from preprocess import preprocess_dataframe


class DialogueSummarizationDataset(Dataset):
    """
    PyTorch Dataset for dialogue summarization.
    
    This dataset wraps preprocessed dialogues and summaries, handling
    tokenization and formatting for seq2seq models.
    
    Args:
        dialogues (list): List of dialogue strings
        summaries (list): List of summary strings
        tokenizer: Hugging Face tokenizer instance
        max_input_length (int): Maximum length for input sequences
        max_target_length (int): Maximum length for target sequences
    """
    
    def __init__(
        self,
        dialogues: list,
        summaries: list,
        tokenizer,
        max_input_length: int = 512,
        max_target_length: int = 128
    ):
        self.dialogues = dialogues
        self.summaries = summaries
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
    
    def __len__(self) -> int:
        return len(self.dialogues)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single training example.
        
        Returns tokenized inputs and labels without padding.
        Padding is handled dynamically by DataCollatorForSeq2Seq during batching.
        
        Returns:
            dict: Dictionary with input_ids, attention_mask, and labels
        """
        dialogue = str(self.dialogues[idx])
        summary = str(self.summaries[idx])
        
        # Tokenize input dialogue without padding (dynamic padding done by collator)
        model_inputs = self.tokenizer(
            dialogue,
            max_length=self.max_input_length,
            truncation=True,
        )
        
        # Tokenize target summary without padding
        labels = self.tokenizer(
            summary,
            max_length=self.max_target_length,
            truncation=True,
        )
        
        # Prepare the item
        # DataCollatorForSeq2Seq will handle padding dynamically
        item = {
            'input_ids': model_inputs['input_ids'],
            'attention_mask': model_inputs['attention_mask'],
            'labels': labels['input_ids']
        }
        
        return item


def initialize_model(model_name: str = "t5-small"):
    """
    Initialize the summarization model and tokenizer.
    
    Loads a pretrained seq2seq model from Hugging Face model hub.
    Supports models like BART, T5, and other encoder-decoder architectures.
    Default is t5-small for faster training and lower memory requirements.
    
    Args:
        model_name (str): Hugging Face model identifier
                         (e.g., 'facebook/bart-large-cnn', 't5-small')
        
    Returns:
        tuple: (model, tokenizer) instances
        
    Raises:
        ValueError: If model_name is not a valid model identifier
    """
    print(f"Loading model: {model_name}")
    
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Load model for seq2seq generation
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
        print(f"✓ Model loaded successfully")
        print(f"  Model type: {model.config.model_type}")
        print(f"  Vocab size: {tokenizer.vocab_size}")
        
        return model, tokenizer
        
    except Exception as e:
        raise ValueError(f"Failed to load model '{model_name}': {str(e)}")


def load_and_preprocess_data(
    data_path: str,
    is_training: bool = True
) -> pd.DataFrame:
    """
    Load and preprocess data for training or evaluation.
    
    Integrates data_loader and preprocess modules to provide clean,
    ready-to-use data for model training.
    
    Args:
        data_path (str): Path to CSV file with dialogue-summary pairs
        is_training (bool): Whether this is training data (affects validation)
        
    Returns:
        pd.DataFrame: Preprocessed DataFrame with cleaned dialogues and summaries
    """
    print(f"Loading data from: {data_path}")
    
    # Load data using data_loader
    df = load_train_data(data_path)
    
    # Preprocess using preprocess module
    df_clean = preprocess_dataframe(df)
    
    print(f"✓ Loaded and preprocessed {len(df_clean)} samples")
    
    return df_clean


def train(
    model_name: str = "t5-small",
    train_data_path: str = "../data/samsum_train.csv",
    val_data_path: Optional[str] = None,
    output_dir: str = "../models/",
    batch_size: int = 4,
    num_epochs: int = 3,
    learning_rate: float = 2e-5,
    max_input_length: int = 512,
    max_target_length: int = 128,
    warmup_steps: int = 500,
    save_steps: int = 1000,
    logging_steps: int = 100
):
    """
    Main training function using Hugging Face Trainer API.
    
    This function orchestrates the complete training pipeline:
    - Loads and preprocesses data
    - Initializes model and tokenizer
    - Sets up training arguments
    - Trains the model using Trainer API
    - Saves the fine-tuned model
    
    Args:
        model_name (str): Pretrained model identifier (t5-small, bart-large-cnn, etc.)
                         Default is t5-small for faster training and lower memory
        train_data_path (str): Path to training CSV file
        val_data_path (str, optional): Path to validation CSV file
        output_dir (str): Directory to save model checkpoints
        batch_size (int): Training batch size (configurable for resources)
        num_epochs (int): Number of training epochs
        learning_rate (float): Learning rate for optimizer
        max_input_length (int): Maximum length for input sequences
        max_target_length (int): Maximum length for target sequences
        warmup_steps (int): Number of warmup steps for learning rate scheduler
        save_steps (int): Save checkpoint every N steps
        logging_steps (int): Log metrics every N steps
    """
    print("=" * 70)
    print("DIALOGUE SUMMARIZATION TRAINING")
    print("=" * 70)
    print()
    
    # Check device availability (GPU or CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print()
    
    # Initialize model and tokenizer
    model, tokenizer = initialize_model(model_name)
    model.to(device)
    print()
    
    # Load and preprocess training data
    train_df = load_and_preprocess_data(train_data_path, is_training=True)
    train_dialogues = train_df['dialogue'].tolist()
    train_summaries = train_df['summary'].tolist()
    print()
    
    # Load validation data if provided
    val_dataset = None
    if val_data_path:
        val_df = load_and_preprocess_data(val_data_path, is_training=False)
        val_dialogues = val_df['dialogue'].tolist()
        val_summaries = val_df['summary'].tolist()
        val_dataset = DialogueSummarizationDataset(
            val_dialogues,
            val_summaries,
            tokenizer,
            max_input_length,
            max_target_length
        )
        print()
    
    # Create training dataset
    train_dataset = DialogueSummarizationDataset(
        train_dialogues,
        train_summaries,
        tokenizer,
        max_input_length,
        max_target_length
    )
    
    print(f"Training samples: {len(train_dataset)}")
    if val_dataset:
        print(f"Validation samples: {len(val_dataset)}")
    print()
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Define training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch" if val_dataset else "no",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        weight_decay=0.01,
        save_total_limit=2,
        save_steps=save_steps,
        logging_steps=logging_steps,
        warmup_steps=warmup_steps,
        predict_with_generate=True,
        fp16=False,  # Disable FP16 by default for compatibility with all GPUs
        push_to_hub=False,  # Cloud-safe: don't push to hub
        report_to="none",  # Cloud-safe: disable external reporting
    )
    
    # Data collator for dynamic padding
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True
    )
    
    # Initialize Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    print("Starting training...")
    print("-" * 70)
    
    # Train the model
    train_result = trainer.train()
    
    print()
    print("-" * 70)
    print("Training completed!")
    print(f"  Final training loss: {train_result.training_loss:.4f}")
    print()
    
    # Save the fine-tuned model and tokenizer
    print(f"Saving model to: {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print()
    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Model saved to: {output_dir}")
    print(f"Total epochs: {num_epochs}")
    print(f"Training samples: {len(train_dataset)}")
    
    return trainer


if __name__ == "__main__":
    """
    Main execution block for training.
    
    This block demonstrates typical usage with configurable parameters.
    Adjust batch_size and max_length based on available GPU memory.
    """
    
    # Configuration (adjust based on your resources)
    config = {
        'model_name': 't5-small',  # Use t5-small for faster training, bart-large-cnn for better quality
        'train_data_path': '../data/samsum_train.csv',
        'val_data_path': '../data/samsum_test.csv',  # Optional: set to None to skip validation
        'output_dir': '../models/dialogue-summarizer/',
        'batch_size': 4,  # Reduce if out of memory, increase if you have more GPU memory
        'num_epochs': 3,
        'learning_rate': 2e-5,
        'max_input_length': 512,  # Maximum dialogue length
        'max_target_length': 128,  # Maximum summary length
        'warmup_steps': 500,
        'save_steps': 1000,
        'logging_steps': 100,
    }
    
    # Run training
    train(**config)
