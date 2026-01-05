"""
Training Module

This module contains the training logic for dialogue summarization models,
including model initialization, training loop, checkpoint management,
and evaluation metrics for fine-tuning transformer models (T5, BART, PEGASUS).
"""

import os
import warnings
from pathlib import Path
from typing import Optional, Dict, Any, Union
import pandas as pd
import numpy as np

import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)

try:
    from datasets import Dataset, DatasetDict
    HF_DATASETS_AVAILABLE = True
except ImportError:
    HF_DATASETS_AVAILABLE = False
    warnings.warn("datasets library not available. Install with: pip install datasets")
    # Define placeholders for type hints
    Dataset = 'Dataset'
    DatasetDict = 'DatasetDict'

try:
    import evaluate
    EVALUATE_AVAILABLE = True
except ImportError:
    EVALUATE_AVAILABLE = False
    warnings.warn("evaluate library not available. Install with: pip install evaluate")

try:
    from .data_loader import DialogueDataLoader
    from .preprocess import DialoguePreprocessor
except ImportError:
    # For direct script execution
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from data_loader import DialogueDataLoader
    from preprocess import DialoguePreprocessor


class DialogueSummarizationTrainer:
    """
    Handles training of dialogue summarization models using Hugging Face Transformers.
    
    Supports T5, BART, PEGASUS, and similar seq2seq models for dialogue summarization.
    Provides complete training pipeline with evaluation metrics, checkpointing, and
    easy integration with preprocessed data.
    """
    
    def __init__(
        self,
        model_name: str = "facebook/bart-base",
        output_dir: str = "models/dialogue-summarization",
        max_input_length: int = 512,
        max_target_length: int = 128,
        dialogue_format: str = "standard"
    ):
        """
        Initialize the trainer.
        
        Args:
            model_name: Pretrained model name or path (e.g., 't5-small', 'facebook/bart-base')
            output_dir: Directory to save model checkpoints
            max_input_length: Maximum input sequence length
            max_target_length: Maximum target sequence length
            dialogue_format: Format for dialogue preprocessing ('standard', 'turns', 'compact')
        """
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
        self.dialogue_format = dialogue_format
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print("="*60)
        print("Initializing Dialogue Summarization Trainer")
        print("="*60)
        print(f"Model: {model_name}")
        
        # Initialize tokenizer and model
        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        print("Loading model...")
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
        # Detect model type for specific handling
        self.model_type = self._detect_model_type()
        print(f"Model type: {self.model_type}")
        
        # Initialize preprocessor
        self.preprocessor = DialoguePreprocessor(
            tokenizer=self.tokenizer,
            max_input_length=max_input_length,
            max_target_length=max_target_length,
            dialogue_format=dialogue_format
        )
        
        # Initialize metrics
        self.metric = None
        if EVALUATE_AVAILABLE:
            try:
                self.metric = evaluate.load("rouge")
                print("ROUGE metrics loaded")
            except Exception as e:
                warnings.warn(f"Could not load ROUGE metrics: {e}")
        
        print(f"Model parameters: {self.model.num_parameters():,}")
        print(f"Output directory: {self.output_dir}")
        print("="*60 + "\n")
    
    def _detect_model_type(self) -> str:
        """
        Detect the model architecture type.
        
        Returns:
            Model type string ('t5', 'bart', 'pegasus', 'other')
        """
        model_name_lower = self.model_name.lower()
        
        if 't5' in model_name_lower:
            return 't5'
        elif 'bart' in model_name_lower:
            return 'bart'
        elif 'pegasus' in model_name_lower:
            return 'pegasus'
        else:
            return 'other'
    
    def load_and_prepare_data(
        self,
        train_df: pd.DataFrame,
        val_df: Optional[pd.DataFrame] = None,
        test_df: Optional[pd.DataFrame] = None,
        val_split: float = 0.1,
        dialogue_col: str = 'dialogue',
        summary_col: str = 'summary'
    ) -> 'DatasetDict':
        """
        Load DataFrames, preprocess, and convert to HF datasets.
        
        Args:
            train_df: Training DataFrame
            val_df: Validation DataFrame (if None, will split from train_df)
            test_df: Test DataFrame (optional)
            val_split: Validation split ratio if val_df is None
            dialogue_col: Name of dialogue column
            summary_col: Name of summary column
            
        Returns:
            DatasetDict with preprocessed and tokenized data
        """
        if not HF_DATASETS_AVAILABLE:
            raise ImportError("datasets library required. Install with: pip install datasets")
        
        print("\n" + "="*60)
        print("Loading and Preparing Data")
        print("="*60)
        
        # Preprocess DataFrames
        print("\nPreprocessing training data...")
        train_processed = self.preprocessor.preprocess_dataframe(
            train_df,
            dialogue_col=dialogue_col,
            summary_col=summary_col
        )
        
        # Split if validation not provided
        if val_df is None and val_split > 0:
            print(f"\nSplitting {val_split*100:.0f}% for validation...")
            from data_loader import DialogueDataLoader
            loader = DialogueDataLoader()
            train_processed, val_processed = loader.split_train_validation(
                train_processed,
                val_split=val_split
            )
        elif val_df is not None:
            print("\nPreprocessing validation data...")
            val_processed = self.preprocessor.preprocess_dataframe(
                val_df,
                dialogue_col=dialogue_col,
                summary_col=summary_col
            )
        else:
            val_processed = None
        
        # Convert to HF datasets
        print("\nConverting to HuggingFace datasets...")
        dataset_dict = {}
        
        dataset_dict['train'] = Dataset.from_pandas(train_processed, preserve_index=False)
        print(f"Train: {len(dataset_dict['train'])} samples")
        
        if val_processed is not None:
            dataset_dict['validation'] = Dataset.from_pandas(val_processed, preserve_index=False)
            print(f"Validation: {len(dataset_dict['validation'])} samples")
        
        if test_df is not None:
            print("\nPreprocessing test data...")
            test_processed = self.preprocessor.preprocess_dataframe(
                test_df,
                dialogue_col=dialogue_col,
                summary_col=summary_col
            )
            dataset_dict['test'] = Dataset.from_pandas(test_processed, preserve_index=False)
            print(f"Test: {len(dataset_dict['test'])} samples")
        
        # Tokenize datasets
        print("\nTokenizing datasets...")
        dataset_dict = DatasetDict(dataset_dict)
        tokenized = self.tokenize_datasets(dataset_dict)
        
        print("="*60 + "\n")
        return tokenized
    
    def tokenize_datasets(
        self,
        dataset_dict: 'DatasetDict',
        num_proc: Optional[int] = None
    ) -> 'DatasetDict':
        """
        Tokenize datasets for training.
        
        Args:
            dataset_dict: DatasetDict containing train/validation/test splits
            num_proc: Number of processes for parallel preprocessing
            
        Returns:
            Tokenized DatasetDict
        """
        processed = DatasetDict()
        
        for split, dataset in dataset_dict.items():
            print(f"Tokenizing {split} split ({len(dataset)} samples)...")
            processed[split] = self.preprocessor.preprocess_dataset(
                dataset,
                batched=True,
                num_proc=num_proc
            )
        
        return processed
    
    def compute_metrics(self, eval_pred) -> Dict[str, float]:
        """
        Compute ROUGE metrics during evaluation.
        
        Args:
            eval_pred: Predictions and labels from evaluation
            
        Returns:
            Dictionary of metric scores
        """
        predictions, labels = eval_pred
        
        # Decode predictions
        decoded_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
        
        # Replace -100 in labels (used for padding)
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        # Clean up text
        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [label.strip() for label in decoded_labels]
        
        # Compute ROUGE scores
        if self.metric is not None:
            result = self.metric.compute(
                predictions=decoded_preds,
                references=decoded_labels,
                use_stemmer=True
            )
            
            # Extract specific scores
            result = {
                'rouge1': result['rouge1'],
                'rouge2': result['rouge2'],
                'rougeL': result['rougeL'],
                'rougeLsum': result['rougeLsum']
            }
            
            return {k: round(v, 4) for k, v in result.items()}
        else:
            return {}
    
    def get_training_arguments(
        self,
        learning_rate: float = 2e-5,
        num_train_epochs: int = 3,
        batch_size: int = 8,
        gradient_accumulation_steps: int = 1,
        warmup_steps: int = 500,
        eval_steps: Optional[int] = None,
        save_steps: Optional[int] = None,
        logging_steps: int = 100,
        generation_max_length: Optional[int] = None,
        generation_num_beams: int = 4,
        **kwargs
    ) -> Seq2SeqTrainingArguments:
        """
        Create training arguments configuration with sensible defaults.
        
        Args:
            learning_rate: Learning rate for optimizer (2e-5 for BART, 1e-4 for T5)
            num_train_epochs: Number of training epochs
            batch_size: Per-device training batch size
            gradient_accumulation_steps: Gradient accumulation steps
            warmup_steps: Number of warmup steps
            eval_steps: Evaluate every N steps (defaults to 10% of training)
            save_steps: Save checkpoint every N steps (defaults to eval_steps)
            logging_steps: Log every N steps
            generation_max_length: Max length for generation (defaults to max_target_length)
            generation_num_beams: Number of beams for generation
            **kwargs: Additional arguments for Seq2SeqTrainingArguments
            
        Returns:
            Training arguments object
        """
        # Set generation length
        if generation_max_length is None:
            generation_max_length = self.max_target_length
        
        # Calculate eval_steps if not provided (10% of total training steps)
        if eval_steps is None:
            eval_steps = max(100, 500)  # Default to 500
        
        if save_steps is None:
            save_steps = eval_steps
        
        print("\nTraining Configuration:")
        print("-"*60)
        print(f"Learning rate: {learning_rate}")
        print(f"Epochs: {num_train_epochs}")
        print(f"Batch size: {batch_size}")
        print(f"Gradient accumulation: {gradient_accumulation_steps}")
        print(f"Effective batch size: {batch_size * gradient_accumulation_steps}")
        print(f"Warmup steps: {warmup_steps}")
        print(f"Eval steps: {eval_steps}")
        print(f"Save steps: {save_steps}")
        print(f"FP16: {torch.cuda.is_available()}")
        print("-"*60 + "\n")
        
        training_args = Seq2SeqTrainingArguments(
            output_dir=str(self.output_dir),
            
            # Evaluation strategy
            evaluation_strategy="steps",
            eval_steps=eval_steps,
            
            # Save strategy
            save_strategy="steps",
            save_steps=save_steps,
            save_total_limit=3,
            
            # Learning
            learning_rate=learning_rate,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=warmup_steps,
            weight_decay=0.01,
            
            # Generation for evaluation
            predict_with_generate=True,
            generation_max_length=generation_max_length,
            generation_num_beams=generation_num_beams,
            
            # Performance
            fp16=torch.cuda.is_available(),
            dataloader_num_workers=0,  # Set to > 0 if you have issues
            
            # Logging and monitoring
            logging_dir=str(self.output_dir / "logs"),
            logging_steps=logging_steps,
            logging_first_step=True,
            report_to=["tensorboard"],
            
            # Model selection
            load_best_model_at_end=True,
            metric_for_best_model="rouge1" if EVALUATE_AVAILABLE else "eval_loss",
            greater_is_better=True if EVALUATE_AVAILABLE else False,
            
            # Other
            push_to_hub=False,
            remove_unused_columns=True,
            
            **kwargs
        )
        
        return training_args
    
    def train(
        self,
        train_dataset: Union['DatasetDict', Dict],
        training_args: Optional[Seq2SeqTrainingArguments] = None,
        resume_from_checkpoint: Optional[str] = None
    ) -> Seq2SeqTrainer:
        """
        Train the model with automatic evaluation and checkpointing.
        
        Args:
            train_dataset: Preprocessed DatasetDict or dict with train/validation
            training_args: Training arguments (uses defaults if None)
            resume_from_checkpoint: Path to checkpoint to resume from
            
        Returns:
            Trained Seq2SeqTrainer object
        """
        print("\n" + "="*60)
        print("Starting Training")
        print("="*60)
        
        if training_args is None:
            training_args = self.get_training_arguments()
        
        # Create data collator
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model,
            padding=True
        )
        
        # Get train and eval datasets
        if isinstance(train_dataset, dict):
            train_data = train_dataset["train"]
            eval_data = train_dataset.get("validation")
        else:
            train_data = train_dataset["train"]
            eval_data = train_dataset.get("validation")
        
        print(f"\nTrain dataset: {len(train_data)} samples")
        if eval_data:
            print(f"Eval dataset: {len(eval_data)} samples")
        
        # Initialize trainer
        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_data,
            eval_dataset=eval_data,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics if EVALUATE_AVAILABLE else None
        )
        
        # Train the model
        print("\n" + "="*60)
        print("Training in progress...")
        print("Monitor with: tensorboard --logdir", str(self.output_dir / "logs"))
        print("="*60 + "\n")
        
        try:
            if resume_from_checkpoint:
                print(f"Resuming from checkpoint: {resume_from_checkpoint}")
                train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
            else:
                train_result = trainer.train()
            
            # Print training metrics
            print("\n" + "="*60)
            print("Training Completed!")
            print("="*60)
            print(f"Training loss: {train_result.training_loss:.4f}")
            print(f"Training samples/second: {train_result.metrics.get('train_samples_per_second', 0):.2f}")
            print("="*60 + "\n")
            
        except KeyboardInterrupt:
            print("\n\nTraining interrupted by user. Saving current state...")
        except Exception as e:
            print(f"\n\nError during training: {e}")
            raise
        
        # Save final model
        final_model_path = self.output_dir / "final_model"
        print(f"Saving final model to {final_model_path}...")
        trainer.save_model(str(final_model_path))
        self.tokenizer.save_pretrained(str(final_model_path))
        
        # Save training arguments
        training_args_path = final_model_path / "training_args.json"
        with open(training_args_path, 'w') as f:
            import json
            json.dump(training_args.to_dict(), f, indent=2)
        
        print("\nModel saved successfully!\n")
        
        return trainer
    
    def resume_training(self, checkpoint_path: str) -> Seq2SeqTrainer:
        """
        Resume training from a checkpoint.
        
        Args:
            checkpoint_path: Path to the checkpoint directory
            
        Returns:
            Trainer object ready to resume training
        """
        print(f"Resuming training from {checkpoint_path}")
        
        training_args = self.get_training_arguments()
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model
        )
        
        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            tokenizer=self.tokenizer,
            data_collator=data_collator
        )
        
        trainer.train(resume_from_checkpoint=checkpoint_path)
        
        return trainer


def train_model_from_dataframes(
    train_df: pd.DataFrame,
    val_df: Optional[pd.DataFrame] = None,
    model_name: str = "facebook/bart-base",
    output_dir: str = "models/dialogue-summarization",
    num_epochs: int = 3,
    batch_size: int = 8,
    learning_rate: float = 2e-5,
    val_split: float = 0.1,
    **training_kwargs
) -> Seq2SeqTrainer:
    """
    Convenience function to train a model from DataFrames.
    
    Args:
        train_df: Training DataFrame with 'dialogue' and 'summary' columns
        val_df: Validation DataFrame (optional, will split from train if None)
        model_name: Model to use ('facebook/bart-base', 't5-small', etc.)
        output_dir: Directory to save model
        num_epochs: Number of training epochs
        batch_size: Batch size per device
        learning_rate: Learning rate
        val_split: Validation split if val_df is None
        **training_kwargs: Additional training arguments
        
    Returns:
        Trained Seq2SeqTrainer object
    """
    # Initialize trainer
    trainer = DialogueSummarizationTrainer(
        model_name=model_name,
        output_dir=output_dir,
        dialogue_format='standard'
    )
    
    # Load and prepare data
    dataset_dict = trainer.load_and_prepare_data(
        train_df=train_df,
        val_df=val_df,
        val_split=val_split
    )
    
    # Get training arguments
    training_args = trainer.get_training_arguments(
        learning_rate=learning_rate,
        num_train_epochs=num_epochs,
        batch_size=batch_size,
        **training_kwargs
    )
    
    # Train
    trained_trainer = trainer.train(dataset_dict, training_args)
    
    return trained_trainer


def main():
    """
    Main training script demonstrating complete training pipeline.
    
    This example shows how to:
    1. Load dialogue data from CSV files
    2. Preprocess and prepare datasets
    3. Configure training parameters
    4. Train a transformer model for summarization
    5. Save the trained model
    """
    print("="*60)
    print("Dialogue Summarization Training Script")
    print("="*60 + "\n")
    
    # ============================================================
    # Configuration
    # ============================================================
    MODEL_NAME = "facebook/bart-base"  # Options: 't5-small', 'facebook/bart-base', 'google/pegasus-cnn_dailymail'
    DATA_DIR = "data"
    OUTPUT_DIR = "models/dialogue-summarization"
    
    # Training hyperparameters
    NUM_EPOCHS = 3
    BATCH_SIZE = 4  # Reduce if OOM errors
    LEARNING_RATE = 2e-5  # 2e-5 for BART, 1e-4 for T5
    VAL_SPLIT = 0.1
    MAX_INPUT_LENGTH = 512
    MAX_TARGET_LENGTH = 128
    
    print("Configuration:")
    print(f"  Model: {MODEL_NAME}")
    print(f"  Epochs: {NUM_EPOCHS}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Learning rate: {LEARNING_RATE}\n")
    
    # ============================================================
    # Load Data
    # ============================================================
    print("Step 1: Loading data...")
    loader = DialogueDataLoader(data_dir=DATA_DIR)
    train_df, test_df = loader.load_train_test_data()
    
    # For faster testing, use a subset
    # train_df = train_df.head(1000)
    
    # ============================================================
    # Initialize Trainer
    # ============================================================
    print("\nStep 2: Initializing trainer...")
    trainer = DialogueSummarizationTrainer(
        model_name=MODEL_NAME,
        output_dir=OUTPUT_DIR,
        max_input_length=MAX_INPUT_LENGTH,
        max_target_length=MAX_TARGET_LENGTH,
        dialogue_format='standard'
    )
    
    # ============================================================
    # Prepare Data
    # ============================================================
    print("Step 3: Preparing datasets...")
    dataset_dict = trainer.load_and_prepare_data(
        train_df=train_df,
        val_df=None,  # Will auto-split
        val_split=VAL_SPLIT
    )
    
    # ============================================================
    # Configure Training
    # ============================================================
    print("Step 4: Configuring training...")
    training_args = trainer.get_training_arguments(
        learning_rate=LEARNING_RATE,
        num_train_epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        gradient_accumulation_steps=2,  # Effective batch size = BATCH_SIZE * 2
        warmup_steps=500,
        eval_steps=500,
        save_steps=500,
        logging_steps=100
    )
    
    # ============================================================
    # Train Model
    # ============================================================
    print("Step 5: Training model...")
    trained_trainer = trainer.train(dataset_dict, training_args)
    
    # ============================================================
    # Final Evaluation
    # ============================================================
    if 'validation' in dataset_dict:
        print("\nStep 6: Final evaluation...")
        eval_results = trained_trainer.evaluate()
        
        print("\nEvaluation Results:")
        print("="*60)
        for key, value in eval_results.items():
            print(f"  {key}: {value:.4f}")
        print("="*60)
    
    # ============================================================
    # Complete
    # ============================================================
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"\nTrained model saved to: {OUTPUT_DIR}/final_model")
    print(f"To use the model:")
    print(f"  from transformers import AutoTokenizer, AutoModelForSeq2SeqLM")
    print(f"  tokenizer = AutoTokenizer.from_pretrained('{OUTPUT_DIR}/final_model')")
    print(f"  model = AutoModelForSeq2SeqLM.from_pretrained('{OUTPUT_DIR}/final_model')")
    print("\nTo view training logs:")
    print(f"  tensorboard --logdir {OUTPUT_DIR}/logs")
    print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    main()
