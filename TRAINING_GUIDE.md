# Training Guide: Dialogue Summarization

This guide covers training transformer-based models for dialogue summarization using the training module.

## Table of Contents
- [Quick Start](#quick-start)
- [Supported Models](#supported-models)
- [Training Pipeline](#training-pipeline)
- [Configuration Options](#configuration-options)
- [Advanced Usage](#advanced-usage)
- [Monitoring Training](#monitoring-training)
- [Troubleshooting](#troubleshooting)

---

## Quick Start

### 1. Basic Training Script

```python
from src.data_loader import DialogueDataLoader
from src.train import DialogueSummarizationTrainer

# Load data
loader = DialogueDataLoader(data_dir="data")
train_df, test_df = loader.load_train_test_data()

# Initialize trainer
trainer = DialogueSummarizationTrainer(
    model_name="facebook/bart-base",
    output_dir="models/my-model"
)

# Load and prepare datasets
dataset_dict = trainer.load_and_prepare_data(
    train_df=train_df,
    val_split=0.1  # 10% validation split
)

# Train with default settings
trained_trainer = trainer.train(dataset_dict)
```

### 2. One-Line Training

```python
from src.train import train_model_from_dataframes

# Train with defaults
trained_trainer = train_model_from_dataframes(
    train_df=train_df,
    model_name="facebook/bart-base",
    output_dir="models/my-model",
    num_epochs=3,
    batch_size=4
)
```

### 3. Command Line Training

```bash
# Run the built-in training script
python src/train.py
```

---

## Supported Models

### BART Models
**Best for:** General summarization, dialogue understanding
```python
trainer = DialogueSummarizationTrainer(
    model_name="facebook/bart-base",  # 139M params
    # or "facebook/bart-large",      # 406M params
    # or "facebook/bart-large-cnn",  # Pre-trained on CNN/DM
)
```

**Recommended settings:**
- Learning rate: `2e-5`
- Batch size: `4-8`
- Max input length: `512-1024`

### T5 Models
**Best for:** Flexible text-to-text tasks
```python
trainer = DialogueSummarizationTrainer(
    model_name="t5-small",   # 60M params
    # or "t5-base",         # 220M params
    # or "t5-large",        # 770M params
)
```

**Recommended settings:**
- Learning rate: `1e-4`
- Batch size: `4-8`
- Max input length: `512`

### PEGASUS Models
**Best for:** High-quality abstractive summarization
```python
trainer = DialogueSummarizationTrainer(
    model_name="google/pegasus-cnn_dailymail",  # 568M params
)
```

**Recommended settings:**
- Learning rate: `5e-5`
- Batch size: `2-4`
- Max input length: `512-1024`

---

## Training Pipeline

### Complete Pipeline Example

```python
from src.data_loader import DialogueDataLoader
from src.train import DialogueSummarizationTrainer

# ============================================================
# Step 1: Load Data
# ============================================================
loader = DialogueDataLoader(data_dir="data")
train_df, test_df = loader.load_train_test_data()

# Optional: Use subset for faster experimentation
# train_df = train_df.head(1000)

# ============================================================
# Step 2: Initialize Trainer
# ============================================================
trainer = DialogueSummarizationTrainer(
    model_name="facebook/bart-base",
    output_dir="models/dialogue-bart",
    max_input_length=512,
    max_target_length=128,
    dialogue_format='standard'  # 'standard', 'turns', or 'compact'
)

# ============================================================
# Step 3: Prepare Datasets
# ============================================================
dataset_dict = trainer.load_and_prepare_data(
    train_df=train_df,
    val_df=None,      # Auto-split if None
    val_split=0.1,    # 10% for validation
    dialogue_col='dialogue',
    summary_col='summary'
)

# ============================================================
# Step 4: Configure Training
# ============================================================
training_args = trainer.get_training_arguments(
    learning_rate=2e-5,
    num_train_epochs=3,
    batch_size=4,
    gradient_accumulation_steps=2,  # Effective batch size = 8
    warmup_steps=500,
    eval_steps=500,
    save_steps=500,
    logging_steps=100,
    generation_num_beams=4
)

# ============================================================
# Step 5: Train
# ============================================================
trained_trainer = trainer.train(
    train_dataset=dataset_dict,
    training_args=training_args
)

# ============================================================
# Step 6: Evaluate
# ============================================================
eval_results = trained_trainer.evaluate()
print(f"ROUGE-1: {eval_results['eval_rouge1']:.4f}")
print(f"ROUGE-2: {eval_results['eval_rouge2']:.4f}")
print(f"ROUGE-L: {eval_results['eval_rougeL']:.4f}")
```

---

## Configuration Options

### Model Configuration

```python
trainer = DialogueSummarizationTrainer(
    model_name="facebook/bart-base",      # Pretrained model
    output_dir="models/my-model",         # Output directory
    max_input_length=512,                 # Max dialogue tokens
    max_target_length=128,                # Max summary tokens
    dialogue_format='standard'            # Preprocessing format
)
```

### Training Hyperparameters

```python
training_args = trainer.get_training_arguments(
    # Learning
    learning_rate=2e-5,                   # Learning rate
    num_train_epochs=3,                   # Number of epochs
    warmup_steps=500,                     # Warmup steps
    weight_decay=0.01,                    # Weight decay
    
    # Batch size
    batch_size=4,                         # Per-device batch size
    gradient_accumulation_steps=2,        # Accumulation steps
    
    # Evaluation
    eval_steps=500,                       # Evaluate every N steps
    
    # Checkpointing
    save_steps=500,                       # Save every N steps
    save_total_limit=3,                   # Keep only 3 checkpoints
    
    # Logging
    logging_steps=100,                    # Log every N steps
    
    # Generation
    generation_max_length=128,            # Max generation length
    generation_num_beams=4,               # Beam search size
)
```

### Data Preprocessing

```python
# Choose dialogue format
trainer = DialogueSummarizationTrainer(
    dialogue_format='standard'  # Preserves speakers
    # 'turns'                   # Explicit turn markers
    # 'compact'                 # Messages only
)

# Custom data loading
dataset_dict = trainer.load_and_prepare_data(
    train_df=train_df,
    val_df=val_df,              # Provide custom validation
    test_df=test_df,            # Optional test set
    dialogue_col='dialogue',    # Column names
    summary_col='summary'
)
```

---

## Advanced Usage

### 1. Custom Training Loop

```python
from transformers import Seq2SeqTrainer, DataCollatorForSeq2Seq

# Initialize components
trainer_obj = DialogueSummarizationTrainer(
    model_name="facebook/bart-base",
    output_dir="models/custom"
)

# Prepare data
dataset_dict = trainer_obj.load_and_prepare_data(train_df, val_split=0.1)

# Custom data collator
data_collator = DataCollatorForSeq2Seq(
    tokenizer=trainer_obj.tokenizer,
    model=trainer_obj.model,
    padding=True,
    max_length=512
)

# Custom training arguments
from transformers import Seq2SeqTrainingArguments

training_args = Seq2SeqTrainingArguments(
    output_dir="models/custom",
    learning_rate=3e-5,
    num_train_epochs=5,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    evaluation_strategy="steps",
    eval_steps=250,
    save_steps=250,
    logging_steps=50,
    predict_with_generate=True,
    generation_max_length=128,
    generation_num_beams=6,
    fp16=True,
    load_best_model_at_end=True,
    metric_for_best_model="rouge1",
    greater_is_better=True
)

# Train
trainer = Seq2SeqTrainer(
    model=trainer_obj.model,
    args=training_args,
    train_dataset=dataset_dict['train'],
    eval_dataset=dataset_dict['validation'],
    tokenizer=trainer_obj.tokenizer,
    data_collator=data_collator,
    compute_metrics=trainer_obj.compute_metrics
)

trainer.train()
```

### 2. Resume Training from Checkpoint

```python
# Find checkpoint
checkpoint_path = "models/my-model/checkpoint-1000"

# Resume training
trained_trainer = trainer.train(
    train_dataset=dataset_dict,
    training_args=training_args,
    resume_from_checkpoint=checkpoint_path
)
```

### 3. Multi-GPU Training

```python
# Set environment variable before running
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

# Trainer will automatically use all available GPUs
training_args = trainer.get_training_arguments(
    batch_size=4,  # Per GPU
    gradient_accumulation_steps=1
)

# Effective batch size = 4 GPUs × 4 batch size = 16
```

### 4. Mixed Precision Training

```python
# Automatically enabled if CUDA is available
training_args = trainer.get_training_arguments(
    batch_size=8,
    # fp16=True is set automatically
)

# Disable if needed
from transformers import Seq2SeqTrainingArguments
training_args = Seq2SeqTrainingArguments(
    output_dir="models/my-model",
    fp16=False,  # Disable FP16
    # ... other args
)
```

### 5. Custom Metrics

```python
def custom_compute_metrics(eval_pred):
    """Custom metrics including length statistics."""
    predictions, labels = eval_pred
    
    # Decode
    decoded_preds = trainer.tokenizer.batch_decode(
        predictions, skip_special_tokens=True
    )
    labels = np.where(labels != -100, labels, trainer.tokenizer.pad_token_id)
    decoded_labels = trainer.tokenizer.batch_decode(
        labels, skip_special_tokens=True
    )
    
    # Compute ROUGE
    result = trainer.metric.compute(
        predictions=decoded_preds,
        references=decoded_labels,
        use_stemmer=True
    )
    
    # Add length statistics
    pred_lens = [len(pred.split()) for pred in decoded_preds]
    result['gen_len'] = np.mean(pred_lens)
    
    return result

# Use custom metrics
trainer = Seq2SeqTrainer(
    # ... other args
    compute_metrics=custom_compute_metrics
)
```

---

## Monitoring Training

### 1. TensorBoard

```bash
# Start TensorBoard
tensorboard --logdir models/my-model/logs

# Open browser to http://localhost:6006
```

Metrics to watch:
- `train/loss` - Training loss (should decrease)
- `eval/loss` - Validation loss (should decrease)
- `eval/rouge1` - ROUGE-1 score (should increase)
- `eval/rouge2` - ROUGE-2 score (should increase)
- `eval/rougeL` - ROUGE-L score (should increase)

### 2. Console Output

Training progress shows:
```
Epoch 1/3
[=====>...........] 50% | Loss: 2.345 | LR: 2e-05
Step 500/3000 | Eval Loss: 1.876 | ROUGE-1: 0.345
```

### 3. Checkpoint Files

```
models/my-model/
├── checkpoint-500/
│   ├── pytorch_model.bin
│   ├── optimizer.pt
│   ├── scheduler.pt
│   └── trainer_state.json
├── checkpoint-1000/
├── checkpoint-1500/
└── final_model/
    ├── pytorch_model.bin
    ├── config.json
    ├── tokenizer_config.json
    └── training_args.json
```

---

## Troubleshooting

### Out of Memory (OOM) Errors

**Problem:** CUDA out of memory during training

**Solutions:**
```python
# 1. Reduce batch size
training_args = trainer.get_training_arguments(
    batch_size=2,  # Smaller batch
    gradient_accumulation_steps=4  # Compensate
)

# 2. Reduce sequence length
trainer = DialogueSummarizationTrainer(
    max_input_length=256,  # Shorter sequences
    max_target_length=64
)

# 3. Use gradient checkpointing
from transformers import Seq2SeqTrainingArguments
training_args = Seq2SeqTrainingArguments(
    output_dir="models/my-model",
    gradient_checkpointing=True,  # Trade compute for memory
    # ... other args
)
```

### Slow Training

**Problem:** Training takes too long

**Solutions:**
```python
# 1. Use smaller dataset for testing
train_df_subset = train_df.head(1000)

# 2. Reduce evaluation frequency
training_args = trainer.get_training_arguments(
    eval_steps=1000,  # Less frequent evaluation
    save_steps=1000
)

# 3. Use fewer beams during evaluation
training_args = trainer.get_training_arguments(
    generation_num_beams=2  # Faster generation
)

# 4. Use multiple workers
training_args = Seq2SeqTrainingArguments(
    dataloader_num_workers=4,  # Parallel data loading
    # ... other args
)
```

### Poor Performance

**Problem:** Model generates low-quality summaries

**Solutions:**
```python
# 1. Train longer
training_args = trainer.get_training_arguments(
    num_train_epochs=5  # More epochs
)

# 2. Try different learning rate
training_args = trainer.get_training_arguments(
    learning_rate=1e-4  # Higher LR for T5
)

# 3. Use larger model
trainer = DialogueSummarizationTrainer(
    model_name="facebook/bart-large"  # Larger capacity
)

# 4. Use pretrained summarization model
trainer = DialogueSummarizationTrainer(
    model_name="facebook/bart-large-cnn"  # Pretrained on summaries
)

# 5. Try different dialogue format
trainer = DialogueSummarizationTrainer(
    dialogue_format='compact'  # Cleaner input
)
```

### Import Errors

**Problem:** Missing dependencies

**Solution:**
```bash
# Install all required packages
pip install transformers==4.35.0
pip install datasets==2.14.0
pip install evaluate==0.4.1
pip install rouge-score==0.1.2
pip install torch torchvision torchaudio
pip install tensorboard
```

### Checkpoint Issues

**Problem:** Can't resume from checkpoint

**Solutions:**
```python
# 1. Check checkpoint path
import os
checkpoint_path = "models/my-model/checkpoint-1000"
if not os.path.exists(checkpoint_path):
    print("Checkpoint not found!")

# 2. Resume with correct path
trained_trainer = trainer.train(
    train_dataset=dataset_dict,
    resume_from_checkpoint=checkpoint_path
)

# 3. Load from checkpoint manually
from transformers import AutoModelForSeq2SeqLM
model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint_path)
```

---

## Training Tips

### 1. Start Small
- Test with 100-1000 samples first
- Use smaller model (bart-base, t5-small)
- Reduce epochs to 1-2

### 2. Monitor Metrics
- Watch for overfitting (eval loss increases)
- Track ROUGE scores during training
- Use TensorBoard for visualization

### 3. Experiment
- Try different dialogue formats
- Adjust learning rates
- Test different models

### 4. Save Resources
- Use gradient accumulation for larger effective batch sizes
- Enable FP16 training
- Limit checkpoint saves with `save_total_limit`

### 5. Best Practices
- Always use validation set
- Save training arguments with model
- Document hyperparameters
- Version your models

---

## Example Training Configs

### Fast Experimentation
```python
trainer = DialogueSummarizationTrainer(
    model_name="facebook/bart-base",
    dialogue_format='compact'
)

training_args = trainer.get_training_arguments(
    num_train_epochs=1,
    batch_size=8,
    eval_steps=100,
    save_steps=100
)

# Use subset
train_df_small = train_df.head(500)
dataset_dict = trainer.load_and_prepare_data(train_df_small, val_split=0.2)
```

### Production Quality
```python
trainer = DialogueSummarizationTrainer(
    model_name="facebook/bart-large",
    max_input_length=1024,
    max_target_length=150,
    dialogue_format='standard'
)

training_args = trainer.get_training_arguments(
    learning_rate=2e-5,
    num_train_epochs=5,
    batch_size=2,
    gradient_accumulation_steps=8,
    warmup_steps=1000,
    eval_steps=500,
    generation_num_beams=6
)

# Full dataset
dataset_dict = trainer.load_and_prepare_data(train_df, val_split=0.1)
```

### Memory-Constrained
```python
trainer = DialogueSummarizationTrainer(
    model_name="t5-small",
    max_input_length=256,
    max_target_length=64,
    dialogue_format='compact'
)

training_args = trainer.get_training_arguments(
    batch_size=1,
    gradient_accumulation_steps=16,
    gradient_checkpointing=True
)
```

---

## Next Steps

After training:
1. **Evaluate:** Test on validation set
2. **Inference:** Use model for predictions (see `inference.py`)
3. **Deploy:** Integrate into application (see `app.py`)
4. **Monitor:** Track performance in production
5. **Iterate:** Retrain with new data

For inference and deployment, see:
- [INFERENCE_GUIDE.md](INFERENCE_GUIDE.md)
- [README.md](README.md)
