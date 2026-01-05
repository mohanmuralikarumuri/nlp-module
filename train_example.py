"""
Quick Training Example

This script demonstrates how to train a dialogue summarization model
with minimal configuration. Good for testing and experimentation.
"""

import warnings
warnings.filterwarnings('ignore')

from src.data_loader import DialogueDataLoader
from src.train import DialogueSummarizationTrainer

def main():
    """
    Train a dialogue summarization model with default settings.
    """
    print("="*70)
    print("QUICK TRAINING EXAMPLE")
    print("="*70)
    print("\nThis example trains a BART model on a subset of data")
    print("Estimated time: 5-10 minutes on GPU, 30-60 minutes on CPU\n")
    
    # ============================================================
    # Configuration
    # ============================================================
    MODEL_NAME = "facebook/bart-base"
    OUTPUT_DIR = "models/example-model"
    USE_SUBSET = True  # Set to False to train on full dataset
    SUBSET_SIZE = 500
    
    # ============================================================
    # Load Data
    # ============================================================
    print("Loading data...")
    loader = DialogueDataLoader(data_dir="data")
    train_df, test_df = loader.load_train_test_data()
    
    # Use subset for faster training
    if USE_SUBSET:
        print(f"Using subset of {SUBSET_SIZE} samples for quick training")
        train_df = train_df.head(SUBSET_SIZE)
    
    print(f"Training samples: {len(train_df)}")
    
    # ============================================================
    # Initialize Trainer
    # ============================================================
    print("\nInitializing trainer...")
    trainer = DialogueSummarizationTrainer(
        model_name=MODEL_NAME,
        output_dir=OUTPUT_DIR,
        max_input_length=512,
        max_target_length=128,
        dialogue_format='standard'
    )
    
    # ============================================================
    # Prepare Data
    # ============================================================
    print("Preparing datasets...")
    dataset_dict = trainer.load_and_prepare_data(
        train_df=train_df,
        val_split=0.15  # 15% validation
    )
    
    # ============================================================
    # Configure Training
    # ============================================================
    print("Configuring training...")
    training_args = trainer.get_training_arguments(
        learning_rate=2e-5,
        num_train_epochs=3,
        batch_size=4,
        gradient_accumulation_steps=2,  # Effective batch size = 8
        warmup_steps=100,
        eval_steps=50,    # Evaluate frequently
        save_steps=50,    # Save frequently
        logging_steps=25  # Log frequently
    )
    
    # ============================================================
    # Train
    # ============================================================
    print("\nStarting training...")
    print("You can monitor progress with: tensorboard --logdir", OUTPUT_DIR + "/logs")
    print("-"*70)
    
    trained_trainer = trainer.train(
        train_dataset=dataset_dict,
        training_args=training_args
    )
    
    # ============================================================
    # Evaluate
    # ============================================================
    print("\nRunning final evaluation...")
    eval_results = trained_trainer.evaluate()
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print("\nFinal Evaluation Metrics:")
    print("-"*70)
    for key, value in sorted(eval_results.items()):
        if isinstance(value, float):
            print(f"  {key:25s}: {value:.4f}")
        else:
            print(f"  {key:25s}: {value}")
    print("-"*70)
    
    # ============================================================
    # Usage Instructions
    # ============================================================
    print("\nModel saved to:", OUTPUT_DIR + "/final_model")
    print("\nTo use your trained model:")
    print("=" *70)
    print("from transformers import AutoTokenizer, AutoModelForSeq2SeqLM")
    print()
    print(f"tokenizer = AutoTokenizer.from_pretrained('{OUTPUT_DIR}/final_model')")
    print(f"model = AutoModelForSeq2SeqLM.from_pretrained('{OUTPUT_DIR}/final_model')")
    print()
    print("# Generate summary")
    print('dialogue = "Alice: Hi! How are you?\\nBob: Good thanks!"')
    print("inputs = tokenizer(dialogue, return_tensors='pt', max_length=512, truncation=True)")
    print("outputs = model.generate(**inputs, max_length=128, num_beams=4)")
    print("summary = tokenizer.decode(outputs[0], skip_special_tokens=True)")
    print("print(summary)")
    print("="*70)
    
    print("\nNext steps:")
    print("  1. Try inference with: python src/inference.py")
    print("  2. Launch web app with: streamlit run app/app.py")
    print("  3. See full training options in: TRAINING_GUIDE.md")
    print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
    except Exception as e:
        print(f"\n\nError during training: {e}")
        print("\nTroubleshooting tips:")
        print("  - Check that data files exist in data/ directory")
        print("  - Ensure required packages are installed:")
        print("    pip install transformers datasets evaluate rouge-score torch")
        print("  - Reduce batch_size if you encounter OOM errors")
        print("  - See TRAINING_GUIDE.md for more help")
        raise
